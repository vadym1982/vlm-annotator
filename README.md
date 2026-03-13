# VLM-Powered Image Annotation Tool

A robust utility for automated image annotation using Vision-Language Models (VLM) like **Qwen2.5-VL** and **Gemma 3** via Ollama. This tool simplifies the creation of object detection datasets by generating bounding boxes in YOLO format and providing conversion tools for COCO and VIA standards.

## 🚀 Features

* **Automated Annotation**: Uses state-of-the-art VLMs to detect objects and generate coordinates.
* **Flexible Normalization**: Supports fixed scaling (e.g., 1000 for Qwen models) or image-size normalization.
* **Smart Parsing**: Automatically validates and fixes JSON output from models.
* **Multi-Format Export**: Convert your YOLO annotations to **COCO JSON** or **VGG Image Annotator (VIA)** formats.
* **Real-time Visualization**: Optional OpenCV-based preview of detected bounding boxes during the process.

## 🛠️ Project Structure

* `annotate.py`: Core script for running VLM-based annotation.
* `convert_annotation.py`: Utility for format conversion (YOLO -> COCO/VIA).
* `config.py`: Configuration manager that handles environment variables.
* `utils.py` & `io_tools.py`: Helper functions for file management and progress tracking.
* `classes.txt`: A simple text file defining the object classes to be detected.

## 📦 Setup & Installation

1. **Install Dependencies**:
```bash
pip install ollama opencv-python numpy pillow python-dotenv

```


2. **Install Ollama**:
Ensure [Ollama](https://ollama.ai/) is installed and the required models are pulled:
```bash
ollama pull qwen2.5-vl:7b
ollama pull gemma3:4b

```


3. **Configure Environment**:
Create a `.env` file in the root directory (refer to `.env` for defaults):
```env
VLM=qwen2.5-vl:7b
SHOW_BOXES=1
VERBOSE=0
SCALE=1000

```


4. **Define Classes**:
Add your target classes to `classes.txt` (one per line):
```text
ball
player
goal

```



## 🖥️ Usage

### 1. Running Annotations

Process a directory of images to generate YOLO `.txt` files:

```bash
python annotate.py /path/to/your/images --overwrite

```

### 2. Converting Formats

After annotating, you can convert the results into a single JSON file for COCO or VIA:

**For VIA (VGG Image Annotator):**

```bash
python convert_annotation.py /path/to/your/images --format via

```

**For COCO JSON:**

```bash
python convert_annotation.py /path/to/your/images --format coco

```

*Note: The resulting `.json` file will be created in the parent directory of your image folder.*

## ⚙️ How it Works

1. **Prompting**: The tool sends a specialized prompt to the VLM requesting bounding boxes in `[xmin, ymin, xmax, ymax]` format.
2. **Normalization**:
* If `SCALE` is set (e.g., 1000), coordinates are divided by this value.
* If `SCALE` is empty, the tool uses the actual image dimensions (via `PIL` for speed).


3. **Conversion**: The `annotate.py` script automatically converts VLM's absolute/scaled boxes into YOLO's relative center-based format: `(x_center, y_center, width, height)`.

## 📝 Error Logging

Any issues during parsing or model communication are logged to `errors.log` in the project root.