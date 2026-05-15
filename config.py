import os
import dotenv


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(os.path.join(ROOT_DIR, ".env"))
VLM_NAME = os.getenv("VLM") or "qwen2.5vl:7b"
LLM_NAME = os.getenv("LLM") or "gemma3:4b"
SHOW_BOXES = bool(int(os.getenv("SHOW_BOXES") or "0"))
VERBOSE = bool(int(os.getenv("VERBOSE") or "0"))
SCALE = int(os.getenv("SCALE") or "0")
MP_LIMIT = float(os.getenv("MP_LIMIT") or "0.92")
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD") or "0.5")
OVERLAP_W=float(os.getenv("OVERLAP_W") or "0.2")
OVERLAP_H=float(os.getenv("OVERLAP_H") or "0.2")

if SCALE == 0:
    SCALE = None

if not os.path.exists(os.path.join(ROOT_DIR, "classes.txt")):
    raise FileNotFoundError("File not found. Create classes.txt in the project root with one class name per line.")

with open(os.path.join(ROOT_DIR, "classes.txt"), "r") as f:
    CLASSES = f.read().splitlines()
