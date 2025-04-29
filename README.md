# RTSP Vehicle Speed Tracker

This Python application uses an RTSP camera feed to detect vehicles, track them, estimate their speed using the YOLO object detection model, and save snapshots of speeding vehicles.

## Features

*   Connects to an RTSP stream via command-line argument.
*   Detects vehicles (cars, motorcycles, buses, trucks) using YOLOv8.
*   Tracks detected vehicles across frames using ByteTrack.
*   Estimates the speed of tracked vehicles based on **horizontal movement** (requires calibration).
*   Displays the video feed with bounding boxes, tracker IDs, and estimated speeds (in mph).
*   Shows the current processing FPS.
*   Allows selection of compute device (CPU or Apple Silicon GPU/MPS).
*   Saves snapshots of vehicles exceeding a speed threshold (default 7 mph) to a `snapshots` directory.

## Requirements

*   Python 3.8+
*   OpenCV (`opencv-python-headless`)
*   Ultralytics (`ultralytics`)
*   NumPy (`numpy`)
*   scikit-learn (`scikit-learn`)
*   PyTorch (`torch`) - Required for GPU support and by Ultralytics.

## Installation

1.  **Clone the repository or download the files.**
2.  **(Optional but Recommended) Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `ultralytics` package will download the specified YOLO model (default: `yolov8n.pt`) on the first run if it's not found locally.*

## Configuration and Calibration

Configuration is primarily done via command-line arguments when running the script. However, **speed calibration requires editing `speed_tracker.py`**:

1.  **Speed Estimation Parameter (CRITICAL FOR ACCURACY):**
    *   The current speed estimation is based on **horizontal pixel movement** across the frame.
    *   You **MUST** calibrate the `PIXELS_PER_METER_HORIZONTAL` variable inside `speed_tracker.py`.
    *   **How to Calibrate (Example):**
        1.  Identify a section of the road in your camera's view where vehicles move mostly horizontally.
        2.  Measure a known **horizontal** distance in the real world within that section (e.g., the width of a traffic lane, markings on the road - let's say 3.5 meters).
        3.  Run the script (or use a simple image viewer) and measure the corresponding **pixel distance** horizontally across the frame *at the approximate depth/location where you measured the real distance*.
        4.  Calculate `PIXELS_PER_METER_HORIZONTAL = pixel_distance / real_distance_meters`.
        5.  Update the value of `PIXELS_PER_METER_HORIZONTAL` in `speed_tracker.py`.
    *   **Accuracy Limitation:** This method is most accurate when vehicles move parallel to the calibration line. Speed estimation for vehicles moving diagonally, at different depths, or where perspective distortion is high will be less accurate. True accuracy requires proper camera calibration (intrinsic/extrinsic parameters) and perspective transformation.

## Usage

Run the script from your terminal, providing the RTSP URL as the main argument.

**Basic Usage (CPU, default model):**
```bash
python speed_tracker.py YOUR_RTSP_URL_HERE
```
*(Replace `YOUR_RTSP_URL_HERE` with your actual camera feed URL)*

**Using M1/M2 GPU (MPS):**
*(This is often the default if MPS is available)*
```bash
python speed_tracker.py YOUR_RTSP_URL_HERE --device mps
```

**Using a different YOLO model (e.g., medium):**
```bash
python speed_tracker.py YOUR_RTSP_URL_HERE --model yolov8m.pt --device mps
```

**Command-line Arguments:**

*   `rtsp_url`: (Required) The URL of the RTSP camera feed.
*   `--model`: (Optional) Path to the YOLOv8 model file (e.g., `yolov8n.pt`, `yolov8s.pt`). Defaults to `yolov8n.pt`.
*   `--device`: (Optional) Computation device to use ('cpu' or 'mps'). Defaults to 'mps' if available on Apple Silicon, otherwise 'cpu'.

A window will appear showing the video feed with detections, tracks, and speeds. Press 'q' to quit. Snapshots are saved in the `snapshots` directory.

## Model Choice

*   The script defaults to `yolov8n.pt` (nano) via the `--model` argument's default. This is fast but least accurate.
*   You can specify other models like `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt` using the `--model` argument for increased accuracy at the cost of performance. Ensure you have sufficient hardware, especially if using larger models on the CPU.

## GPU Acceleration (Apple Silicon)

*   If you are using a Mac with an M1, M2, or later chip, the script attempts to use the Metal Performance Shaders (MPS) backend via PyTorch for GPU acceleration by default.
*   You can explicitly select the device using `--device mps` or `--device cpu`.
*   Using MPS (`--device mps`) significantly improves performance (FPS) compared to CPU (`--device cpu`).

## Limitations

*   **Speed Accuracy:** Highly dependent on the calibration of `PIXELS_PER_METER_HORIZONTAL`. Requires careful tuning and is inherently limited without full camera calibration and perspective correction.
*   **Performance:** Processing speed (FPS) depends on hardware (CPU/GPU), YOLO model size, stream resolution, and network conditions.
*   **RTSP Stream Stability:** Relies on a stable network connection and RTSP source. Basic reconnection logic is included.
*   **Tracking:** Tracking might fail or switch IDs in complex scenarios (occlusions, vehicles stopping/starting quickly, high density).
