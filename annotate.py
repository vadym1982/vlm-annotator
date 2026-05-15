import argparse
import datetime
import json
import os
from time import time
import cv2
import numpy as np
import ollama
import config
from detector import detect, detect_sahi
from io_tools import print_progress, draw_boxes
from utils import get_images



log_file = os.path.join(config.ROOT_DIR, "errors.log")

with open(log_file, "w") as _:
    pass

def main():
    parser = argparse.ArgumentParser(description="VLM Image Annotator")

    parser.add_argument(
        "image_dir",
        type=str,
        help="Path to directory with images to annotate"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt annotations. If not set, existing files will be skipped."
    )

    parser.add_argument(
        "--sahi",
        nargs=2,
        type=int,
        metavar=('ROWS', 'COLS'),
        help="Enable SAHI with specified number of rows and columns (e.g., --sahi 2 2). If not set, SAHI is disabled."
    )

    args = parser.parse_args()
    path_to_images = args.image_dir
    overwrite = args.overwrite

    if args.sahi:
        sahi_rows, sahi_cols = args.sahi
    else:
        sahi_rows, sahi_cols = None, None

    if sahi_rows and sahi_cols:
        print(f"SAHI enabled: {sahi_rows}x{sahi_cols}")

    images = get_images(path_to_images)
    images_count = len(images)
    time_per_image = []

    for img_num, image_path in enumerate(images):
        t0 = time()
        annotation_path = f"{os.path.splitext(image_path)[0]}.txt"

        # --- Skipping already annotated image if `overwrite` is False ---

        if os.path.exists(annotation_path) and (not overwrite):
            if config.VERBOSE:
                print(f"Annotation for {image_path} already exists {img_num + 1}/{images_count}")
            else:
                print_progress(img_num + 1, images_count, "?")
            continue

        # --- Reading image ---

        img = cv2.imread(image_path)

        if config.VERBOSE:
            print(f"Annotating image {os.path.basename(image_path)} {img_num + 1}/{images_count} [{(img_num + 1) / images_count * 100:0.2f} %]")

        # --- Detection ---

        t1 = time()

        if sahi_rows and sahi_cols:
            boxes, errors, response = detect_sahi(
                img,
                sahi_rows,
                sahi_cols,
                overlap_w=config.OVERLAP_W,
                overlap_h=config.OVERLAP_H,
                nms_threshold=config.NMS_THRESHOLD
            )
        else:
            boxes, errors, response = detect(img)

        if len(errors) is None:
            with open(log_file, "a") as f:
                f.write(f"ERROR when annotating {image_path}\n")
                f.write(f"E: {errors}\n")

                if len(response) > 0:
                    f.write(f"Response: {response['response']}\n")

        time_per_image.append(time() - t0)
        eta = np.mean(time_per_image) * (images_count - img_num - 1)
        eta_str = str(datetime.timedelta(seconds=round(eta)))

        if config.VERBOSE:
            print(f"  * annotation time for image: {time() - t1:0.2f}s")
            print(f"  * ETA: {eta_str}")

        # --- Saving annotation in YOLOv4 txt format ---

        lines = []

        for cls_id, cls in enumerate(config.CLASSES):
            if cls not in boxes:
                continue

            for box in boxes[cls]:
                try:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    x, y, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    lines.append(f"{cls_id} {x:0.5f} {y:0.5f} {bw:0.5f} {bh:0.5f}")
                except Exception as e:
                    with open(log_file, "a") as f:
                        f.write(f"ERROR when annotating {image_path}\n")
                        f.write(f"E: {e}\n")
                        f.write(f"Boxes: {boxes}\n")

        text = "\n".join(lines)

        with open(annotation_path, "w") as f:
            f.write(text)

        if not config.VERBOSE:
            print_progress(img_num + 1, images_count, eta_str)

        # --- Visualization ---

        try:
            if config.SHOW_BOXES:
                height, width = img.shape[:2]
                mp = width * height / 1e6

                if mp > config.MP_LIMIT:
                    factor = np.sqrt(config.MP_LIMIT / mp)
                    out = cv2.resize(img, (int(width * factor), int(height * factor)))
                else:
                    out = img

                draw_boxes(out, boxes, 1, 1)
                cv2.imshow("annotation", out)
                cv2.waitKey(1)
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"ERROR when annotating {image_path}\n")
                f.write(f"E: {e}\n")
                f.write(f"Boxes: {boxes}\n")

if __name__ == "__main__":
    main()