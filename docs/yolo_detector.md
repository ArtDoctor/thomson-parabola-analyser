# YOLO Region of Interest Detector

To improve the accuracy of line detection and prevent noise from the edges of the Image Plate or detector casing from corrupting the analysis, the Oblisk pipeline employs a YOLO-based region of interest (ROI) detector.

## Purpose

Raw experimental images often contain borders, timestamps, or physical artifacts far away from the actual spectrometer data. The YOLO model's job is to predict a bounding box that encapsulates the active phosphor screen or data area, allowing the pipeline to crop the image before running expensive or sensitive algorithms.

## Model Details

- **Model File:** `yolo-tune/thomson-cutter.onnx`
- **Architecture:** YOLO (Ultralytics) trained for object detection.
- **Classes:** Currently trained to detect a single class (the main Thomson parabola region). See `yolo-tune/classes.txt` (typically `main`).
- **Format:** Exported to ONNX for lightweight, CPU-friendly inference via `onnxruntime` without requiring heavy PyTorch dependencies in the main path.

## Integration in the Pipeline

Located in `oblisk/runtime_yolo.py`:

- **Function:** `cut_detector_image(image: np.ndarray) -> Coords`
- **Execution:** 
  1. The input image (converted to 3-channel BGR if necessary) is passed to the ONNX session.
  2. The model predicts bounding boxes with confidence scores.
  3. The bounding box with the highest confidence (above the 0.5 threshold) is selected.
  4. The coordinates `[x1, y1, x2, y2]` are returned and used to crop the image array.
- **Fallback:** If no bounding box is detected or the confidence is too low, the function safely returns the coordinates of the entire original image, effectively skipping the crop step.
