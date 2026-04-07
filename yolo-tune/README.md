# YOLO Detector Training Assets

This directory keeps the detector-training side of the project together with the exported runtime model used by the main pipeline.

## What lives here

- `thomson-cutter.onnx`: exported detector model used by the Oblisk runtime
- `data.yaml`: Ultralytics dataset config (single class `main`, id `0`)
- `classes.txt`: human-readable class name (kept in sync with `data.yaml`)
- `data/`: raw detector TIFFs (optional staging)
- `data-png/`: PNG output from `tif_to_png.py` (default target)
- `images/` and `labels/`: YOLO layout — paired `*.png` / `*.txt` with the same basename; training uses `images/train`, `images/val`, `labels/train`, `labels/val`
- `train.ipynb`: split train/val, train with Ultralytics, export ONNX into `thomson-cutter.onnx`
- `tif_to_png.py`: convert TIFF inputs into 8-bit PNG for labeling or training

## Runtime relationship

The main analysis pipeline loads `thomson-cutter.onnx` from this directory by default. That path is configurable through `OBLISK_YOLO_MODEL_PATH`.

---

## Labeling and preparing data

Use label-studio. Upload experimental/synthetic images, and then label them. Export them in default YOLO format, copy all files into the yolo-tune folder.


## Training the model

1. **Environment**  
   Use the project virtualenv at the repo root (`venv/`) and install training dependencies there, for example:

   ```bash
   ./venv/bin/pip install ultralytics onnx
   ```

   PyTorch with CUDA is recommended if you have a GPU; follow the [PyTorch install](https://pytorch.org/get-started/locally/) instructions for your platform.

2. **Open `train.ipynb`**  
   Set the notebook kernel to that `venv`. **Start Jupyter or VS Code’s notebook UI with cwd = `yolo-tune`** (or `cd yolo-tune` before launching) so `data.yaml` and relative paths work.

3. **Run the split cell**  
   Populates `images/train`, `images/val`, `labels/train`, `labels/val` from the flat `images/` + `labels/` pairs.

4. **Run the training cell**  
   Trains with Ultralytics (`YOLO('yolo26s.pt')`, `data='data.yaml'`, etc.). Weights and logs go under a `runs/` directory (exact folder name includes the `name=` argument and Ultralytics’ run indexing). Adjust `epochs`, `batch`, `imgsz`, and augmentation flags in the notebook to match your hardware and data.

5. **Export for Oblisk**  
   The notebook loads `best.pt` from the completed run, runs `export(format='onnx', dynamic=True)`, and copies the result to `yolo-tune/thomson-cutter.onnx`. Update the path in the export cell if your run name or `project=` differs from the example.

6. **Sanity check**  
   Run the pipeline on a few real images or use the notebook’s ONNX smoke cell to confirm boxes look reasonable before relying on the new model in production.

---

## TIFF conversion helper (reference)

From the repository root:

```bash
python yolo-tune/tif_to_png.py
```

The converter resolves paths independently of the current working directory and handles both 8-bit and higher-bit-depth TIFF inputs.
