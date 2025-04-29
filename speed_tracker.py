import cv2
import time
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import math
from sklearn.cluster import KMeans
import argparse
import torch

# --- Configuration ---
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes for car, motorcycle, bus, truck
CONFIDENCE_THRESHOLD = 0.5
TRACK_MAX_AGE = 30 # Frames a track can exist without detection

# --- Speed Estimation Parameters (NEEDS CALIBRATION) ---
# This is now configured for HORIZONTAL movement (left/right across the frame).
# You MUST calibrate these based on your camera setup and a known distance.
# Define pixels per meter HORIZONTALLY at a reference line/area.
PIXELS_PER_METER_HORIZONTAL = 53.74 # Example value: Adjust this! Calibrate based on a known horizontal distance.
# REAL_HORIZONTAL_DISTANCE_METERS = 5.0 # e.g., width of a lane
# PIXEL_HORIZONTAL_DISTANCE = 100 # Pixel distance corresponding to REAL_HORIZONTAL_DISTANCE_METERS

# --- OLD VERTICAL CALIBRATION (Commented Out) ---
# PIXELS_PER_METER_VERTICAL = 53.74 # Calibrated from 1.365 pixels/inch (1.365 * 39.3701)

# --- Class Name Mapping ---
CLASS_NAMES = { # Based on COCO indices
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# --- Initialization ---
def main(rtsp_url, model_path, device):
    model = YOLO(model_path)
    try:
        model.to(device)
    except Exception as e:
        print(f"Warning: Could not move model to device '{device}'. Error: {e}. Defaulting to CPU.")
        device = 'cpu'
        model.to(device)

    track_history = defaultdict(lambda: [])
    last_seen_time = defaultdict(float)
    estimated_speeds = defaultdict(float)
    snapshot_saved_ids = set() # Keep track of IDs for which snapshot was saved
    SNAPSHOT_DIR = "snapshots"

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream: {rtsp_url}")
        exit()

    print("Successfully connected to RTSP stream.")
    frame_count = 0
    last_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. Reconnecting attempt or stream end?")
            time.sleep(1) # Wait before retrying or exiting
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print("Reconnection failed. Exiting.")
                break
            else:
                print("Reconnected.")
                continue

        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        fps = 1.0 / delta_time if delta_time > 0 else 0

        results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=CONFIDENCE_THRESHOLD, tracker="bytetrack.yaml", device=device, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            annotated_frame = results[0].plot()

            frame_height, frame_width, _ = annotated_frame.shape
            left_bound = frame_width * 0.20
            right_bound = frame_width * 0.80

            for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                point = (center_x, center_y)

                track_history[track_id].append(point)
                last_seen_time[track_id] = current_time

                if len(track_history[track_id]) > TRACK_MAX_AGE:
                    track_history[track_id].pop(0)

                speed_mph = 0.0
                is_in_center_zone = left_bound <= center_x <= right_bound
                if len(track_history[track_id]) >= 2 and is_in_center_zone:
                    prev_point = track_history[track_id][-2]
                    curr_point = track_history[track_id][-1]

                    pixel_distance_x = abs(curr_point[0] - prev_point[0])

                    real_distance_meters = pixel_distance_x / PIXELS_PER_METER_HORIZONTAL

                    if delta_time > 0:
                        speed_mps = real_distance_meters / delta_time
                        estimated_speeds[track_id] = speed_mps

                    current_speed_mps = estimated_speeds.get(track_id, 0.0)
                    speed_mph = current_speed_mps * 2.23694

                # --- Draw Speed Info FIRST ---
                label = f"ID:{track_id}"
                if speed_mph > 0.1: # Only add speed if it was calculated and is non-trivial
                    label += f" {speed_mph:.1f} mph"

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # Draw the rectangle and text onto the annotated_frame
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # --- Save Snapshot --- (If in center zone, not already saved, AND speed > 7 MPH)
                if is_in_center_zone and track_id not in snapshot_saved_ids:
                    # Check speed *before* deciding to save (using the already calculated speed_mph)
                    if speed_mph > 7.0: # Check the calculated mph speed
                        # Create directory if it doesn't exist
                        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                        # Generate filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(SNAPSHOT_DIR, f"{timestamp}_snapshot_id{track_id}.jpg") # <-- Timestamp first
                        # Save the frame (which now includes the text)
                        cv2.imwrite(filename, annotated_frame) # <-- Save the frame with text drawn
                        # Log details
                        vehicle_type = CLASS_NAMES.get(cls_id, 'unknown')
                        # Extract ROI from the *original* frame for color analysis
                        roi = frame[y1:y2, x1:x2]
                        dominant_bgr = get_dominant_color(roi)
                        color_name = map_bgr_to_name(dominant_bgr)

                        print(f"Snapshot Saved: {filename} | ID: {track_id}, Type: {vehicle_type}, Color: {color_name}, Speed: {speed_mph:.1f} mph") # <-- Use speed_mph here too
                        # Mark as saved
                        snapshot_saved_ids.add(track_id)

        else:
            annotated_frame = frame.copy()

        fps_text = f"FPS: {fps:.2f}"
        font_scale = 2.0
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        frame_width = annotated_frame.shape[1]
        top_right_x = frame_width - text_w - 10
        top_right_y = text_h + 10
        cv2.putText(annotated_frame, fps_text, (top_right_x, top_right_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        cv2.imshow("Vehicle Speed Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream released and windows closed.")

def get_dominant_color(roi, k=3):
    if roi is None or roi.shape[0] < 2 or roi.shape[1] < 2:
        return (0, 0, 0)

    try:
        pixels = roi.reshape((-1, 3))
        pixels = np.float32(pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        _, counts = np.unique(labels, return_counts=True)
        dominant_center = centers[np.argmax(counts)]

        return tuple(map(int, dominant_center))
    except cv2.error as e:
        print(f"OpenCV Error in K-means: {e}")
        return (0, 0, 0)
    except Exception as e:
        print(f"Error in get_dominant_color: {e}")
        return (0, 0, 0)

def color_distance(c1, c2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(c1, c2)]))

def map_bgr_to_name(bgr_color):
    colors = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'gray': (128, 128, 128),
        'silver': (192, 192, 192),
    }
    min_dist = float('inf')
    closest_color = 'unknown'
    for name, ref_bgr in colors.items():
        dist = color_distance(bgr_color, ref_bgr)
        intensity = sum(bgr_color) / 3
        if intensity < 50 and dist > color_distance(bgr_color, colors['black']):
             dist = color_distance(bgr_color, colors['black'])
             name = 'black'
        elif intensity > 200 and dist > color_distance(bgr_color, colors['white']):
             dist = color_distance(bgr_color, colors['white'])
             name = 'white'
        elif 70 < intensity < 180 and name not in ['black', 'white', 'gray', 'silver'] and dist > color_distance(bgr_color, colors['gray']):
            gray_dist = color_distance(bgr_color, colors['gray'])
            if gray_dist < min_dist and gray_dist < 100:
                 min_dist = gray_dist
                 closest_color = 'gray'
                 continue

        if dist < min_dist:
            min_dist = dist
            closest_color = name

    intensity = sum(bgr_color) / 3
    if closest_color != 'black' and intensity < 60 and min_dist > 100:
        return 'dark'
    if closest_color != 'white' and intensity > 190 and min_dist > 100:
        return 'light'

    return closest_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Speed Tracking with YOLOv8")
    parser.add_argument("rtsp_url", help="URL of the RTSP camera feed")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to the YOLOv8 model file (e.g., yolov8n.pt, yolov8s.pt)")

    default_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    parser.add_argument(
        "--device",
        default=default_device,
        choices=['cpu', 'mps'],
        help=f"Computation device to use ('cpu' or 'mps'). Default: {default_device}"
    )

    args = parser.parse_args()

    main(args.rtsp_url, args.model, args.device)
