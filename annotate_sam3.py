import argparse
import datetime
import json
import os
from time import time
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Sam3Model  # Assuming Hugging Face API usage
import config
from io_tools import print_progress, draw_boxes
from utils import get_images


def main():
    parser = argparse.ArgumentParser(description="SAM 3 Auto-Annotator")
    parser.add_argument("image_dir", type=str, help="Path to images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels")
    args = parser.parse_args()

    images = get_images(args.image_dir)
    images_count = len(images)
    time_per_image = []
    log_file = os.path.join(config.ROOT_DIR, "sam3_errors.log")

    # --- SAM 3 Initialization ---
    print("[*] Loading SAM 3 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Official model identifier on Hugging Face
    model_id = "facebook/sam3"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    for img_num, image_path in enumerate(images):
        t0 = time()
        annotation_path = f"{os.path.splitext(image_path)[0]}.txt"

        if os.path.exists(annotation_path) and (not args.overwrite):
            continue

        # --- Image Reading and Preparation ---
        img_bgr = cv2.imread(image_path)
        # SAM 3 requires RGB format for processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # --- Detection via Promptable Concept Segmentation (PCS) ---
        # SAM 3 allows passing a list of classes as text prompts
        inputs = processor(images=img_rgb, text=config.CLASSES, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-processing results to obtain bounding boxes
        # SAM 3 returns masks/scores, from which we extract coordinates
        results = processor.post_process_segmentation(outputs, target_sizes=[(h, w)])[0]

        lines = []
        # Iterate through each detected object in the frame
        for i, class_label in enumerate(results["labels"]):
            # Obtain box coordinates in [xmin, ymin, xmax, ymax] format
            box = results["boxes"][i].cpu().numpy()
            cls_id = config.CLASSES.index(class_label) if class_label in config.CLASSES else -1

            if cls_id != -1:
                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x1, y1, x2, y2 = box
                xn, yn = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                wn, hn = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cls_id} {xn:0.5f} {yn:0.5f} {wn:0.5f} {hn:0.5f}")

        # --- Saving Annotations ---
        with open(annotation_path, "w") as f:
            f.write("\n".join(lines))

        # --- Visualization ---
        if config.SHOW_BOXES:
            # Create a dictionary for compatibility with the existing draw_boxes function
            vis_boxes = {config.CLASSES[int(l.split()[0])]: [[float(x) for x in l.split()[1:]]] for l in lines}
            # Since draw_boxes in annotate.py works with absolute pixel values,
            # we pass width and height as scale factors
            draw_boxes(img_bgr, vis_boxes, w, h)
            cv2.imshow("SAM 3 Annotation", img_bgr)
            cv2.waitKey(1)

        time_per_image.append(time() - t0)
        eta = np.mean(time_per_image) * (images_count - img_num - 1)
        print_progress(img_num + 1, images_count, str(datetime.timedelta(seconds=round(eta))))


if __name__ == "__main__":
    main()