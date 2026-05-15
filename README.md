# VLM-Powered Image Annotation Tool

A professional utility for automated image annotation using Vision-Language Models (VLM) such as **Qwen2.5-VL** and **Gemma 3** via Ollama. This tool streamlines the creation of object detection datasets by generating bounding boxes in YOLO format and providing conversion tools for COCO and VIA standards.

## 🚀 Features

* **Automated Annotation**: Leverages state-of-the-art VLMs to detect objects and generate coordinates.
* **SAHI Support (Slicing)**: Enables processing of high-resolution images by slicing them into patches, significantly improving the detection of small objects.
* **Flexible Normalization**: Supports fixed scaling (e.g., 1000 for Qwen models) or image-size normalization.
* **Smart Parsing**: Automatically validates and refines JSON output from models using an additional LLM correction step.
* **Multi-Format Export**: Convert YOLO annotations to **COCO JSON** or **VGG Image Annotator (VIA)** formats.
* **Real-time Visualization**: Optional OpenCV-based preview of detected boxes during the process.

## 🛠️ Project Structure

* `annotate.py`: Core script for running annotations, supporting both standard and SAHI modes.
* `detector.py`: Contains VLM interaction logic, slicing algorithms, and NMS.
* `convert_annotation.py`: Utility for format conversion (YOLO -> COCO/VIA).
* `config.py`: Configuration manager for environment variables and detection thresholds.
* `classes.txt`: Simple text file defining the target object classes.

## 📦 Setup & Installation

1. **Install Dependencies**:
```bash
pip install ollama opencv-python numpy pillow python-dotenv

```


2. **Install Ollama**:
Ensure [Ollama](https://ollama.ai/) is installed and pull the required models:
```bash
ollama pull qwen2.5-vl:7b
ollama pull gemma3:4b

```


3. **Configure Environment**:
Create a `.env` file in the root directory:
```env
VLM=qwen2.5-vl:7b
SHOW_BOXES=1
VERBOSE=0
SCALE=1000
OVERLAP_W=0.2
OVERLAP_H=0.2
NMS_THRESHOLD=0.5

```



## 🖥️ Usage

### 1. Running Annotations

#### Standard Mode (Process whole image):

```bash
python annotate.py /path/to/your/images --overwrite

```

#### SAHI Mode (Slicing):

Use the `--sahi` flag followed by the number of rows and columns:

```bash
# Slice each image into a 2x2 grid (4 patches total)
python annotate.py /path/to/your/images --sahi 2 2 --overwrite

```

### 2. Converting Formats

**For VIA (VGG Image Annotator):**

```bash
python convert_annotation.py /path/to/your/images --format via

```

**For COCO JSON:**

```bash
python convert_annotation.py /path/to/your/images --format coco

```

## ⚙️ How it Works

1. **Slicing (if SAHI is enabled)**: The image is divided into a grid of patches with a specified overlap.
2. **Prompting**: A specialized prompt is sent to the VLM for each patch (or the full image) to retrieve boxes in `[xmin, ymin, xmax, ymax]` format.
3. **Result Merging**:
* Patch-level coordinates are recalculated into global image coordinates.
* **Non-Maximum Suppression (NMS)** is applied to remove duplicate detections at the patch boundaries.


4. **Normalization**: Final coordinates are normalized to the YOLO format: `(x_center, y_center, width, height)` within the $[0, 1]$ range.

## 📝 Error Logging

All parsing errors or model communication failures are recorded in `errors.log` located in the project root.

---