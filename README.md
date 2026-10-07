# Bone Fracture Detection with YOLOv8n

A local website for testing a trained YOLOv8n bone fracture detection model. Upload an image or choose an example to view class labels, confidence scores, and bounding boxes. The page also displays validation metrics from the project report.

All website HTML, CSS, and JavaScript are in **`index.html`**. **`server.py`** serves the page and runs the supplied PyTorch checkpoint on CPU.

## Features

- Image upload using a file picker or drag-and-drop.
- Three supplied example images.
- Adjustable confidence threshold, starting at 25%.
- Original and annotated image views.
- Detection labels, confidence scores, counts, and processing time.
- Downloadable annotated PNG images and JSON results.
- Aggregate and per-class validation metrics.
- Responsive desktop and mobile layout.
- Local inference without an external API.

## Requirements

- Python **3.13 or newer**, as specified in `pyproject.toml`.
- A modern web browser.
- The supplied `boneFractureDetection-best.pt` in the project folder.
- Installed project dependencies: Ultralytics, OpenCV, and their dependencies, including PyTorch and Pillow.

A GPU is not required: the service uses CPU inference. Internet access is needed for initial dependency installation; the supplied model runs locally after setup.

## How to run

### Windows: use the existing environment

Open PowerShell in the project folder (the folder containing `server.py`), then start the service:

```powershell
.\.venv\Scripts\python.exe server.py
```

Wait for the terminal to print `Open http://127.0.0.1:8000`, then open:

**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Keep the terminal running while using the website. Press **Ctrl+C** in that terminal to stop it. You do not need to activate the environment when using the command above.

### Fresh setup with uv

If `uv` is installed, run these commands from the project folder:

```powershell
uv sync --locked
uv run --locked python server.py
```

This creates or updates `.venv` using the versions in `uv.lock`. Open **http://127.0.0.1:8000** after the model loads.

### Alternative setup with pip (Windows)

With Python 3.13 installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install 'ultralytics>=8.4.174' 'opencv-python>=5.0.0.93' 'ipykernel>=7.4.0'
.\.venv\Scripts\python.exe server.py
```

Ultralytics installs its required inference dependencies, including PyTorch and Pillow. This follows the version constraints in `pyproject.toml`; use uv to reproduce the lockfile's versions.

### Use a different port

If port 8000 is already in use:

```powershell
.\.venv\Scripts\python.exe server.py --port 8001
```

Open **[http://127.0.0.1:8001](http://127.0.0.1:8001)** instead. The server binds to `127.0.0.1` and is accessible only from the computer running the server.

## How to test the model

1. Open the website and wait for **Model ready**.
2. Upload a **JPG, PNG, or WebP**, or choose an example.
3. Adjust the confidence threshold if desired.
4. Click **Run detection**.
5. Review the bounding boxes, predicted classes, and confidence scores.
6. Switch between **Original** and **Detections** to compare views.
7. Use **Save annotated image** or **Export results** to download the output.

Uploads must be no larger than **10 MB** and **20 megapixels**. After changing the threshold, click **Run detection** again. Higher thresholds retain only more confident detections. **Reset** clears the selected image and results.

No detections means no regions exceeded the threshold; it does not confirm a healthy bone. Example filenames are descriptive names, not verified ground-truth annotations.

Opening `index.html` directly displays the interface, but **real predictions require `server.py`**. A browser cannot execute the PyTorch `.pt` checkpoint directly. Use `server.py`, rather than the original placeholder `main.py`, to start the website.

## Model classes

The checkpoint uses **YOLOv8n**, a configured inference image size of **640 pixels**, and 10 classes:

| ID | Class |
| --- | --- |
| 0 | Comminuted |
| 1 | Greenstick |
| 2 | Healthy |
| 3 | Linear |
| 4 | Oblique Displaced |
| 5 | Oblique |
| 6 | Segmental |
| 7 | Spiral |
| 8 | Transverse Displaced |
| 9 | Transverse |

## Validation metrics

The website uses the final `results_dict` on **page 27 of `Bone_Fracture_YOLO.pdf`**:

| Metric | Reported value |
| --- | ---: |
| Precision | 84.3785% |
| Recall | 82.7823% |
| mAP@50 | 86.8951% |
| mAP@50-95 | 50.9618% |

The report records **1,347 training images**, **128 validation images**, and **157 validation instances**. Training was configured for up to 100 epochs, stopped early after **57 epochs**, and selected the best model from **epoch 37** (page 14).

- **Precision:** how many predicted detections are correct.
- **Recall:** how many ground-truth objects are recovered.
- **mAP@50:** mean average precision at an intersection-over-union (IoU) threshold of 0.50.
- **mAP@50-95:** mean AP averaged across IoU thresholds from 0.50 to 0.95.

These are fixed validation metrics, separate from individual prediction confidence. They are not recalculated from each uploaded image. New precision or recall measurements require ground-truth annotations. Classification accuracy was not reported. The checkpoint's stored training metrics differ slightly from the PDF's final validation results; the website displays the PDF values.

## Project files

| File or folder | Purpose |
| --- | --- |
| `index.html` | Website with inline HTML, CSS, and JavaScript |
| `server.py` | Local web server and CPU inference |
| `boneFractureDetection-best.pt` | Trained model checkpoint |
| `assets/` | Three example images |
| `Bone_Fracture_YOLO.pdf` | Training and validation report |
| `Bone_Fracture_YOLO.ipynb` | Existing project notebook |
| `main.ipynb` | Existing notebook |
| `main.py` | Original placeholder script |
| `pyproject.toml` | Python requirements and dependencies |
| `uv.lock` | Dependency lockfile |
| `.python-version` | Project Python version |
| `README.md` | Project documentation |

The notebooks are not required to run the website. The service does not execute or modify them, retrain the model, or overwrite its weights.

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `.venv\Scripts\python.exe` is missing | Complete the uv or pip setup above. |
| `ModuleNotFoundError` | Install dependencies in the environment used to run `server.py`. |
| Model checkpoint not found | Place `boneFractureDetection-best.pt` beside `server.py`. |
| Port already in use / `WinError 10048` | Start with `--port 8001` and open that port's URL. |
| **Server offline** | Start `server.py`, check the terminal for errors, then reload the page. |
| **Run detection** is disabled | Select a valid image and wait for **Model ready**. |
| Upload rejected | Use a valid JPG, PNG, or WebP within the size and pixel limits. |
| No detections | Try a lower threshold and run again; review image quality and relevance. |
| Prediction fails or times out | Check the server terminal for details and retry with a smaller image. |

## Local processing and limitations

Uploads are processed in memory and are not saved by the server. Downloads are saved only when requested through the browser. CPU processing time varies by computer and image; the page shows measured model-processing time.

This project is for research, demonstration, and model testing. Predictions are not a clinical diagnosis. An empty result does not establish that a bone is healthy.
