import argparse
import datetime
import json
import os
from time import time
import cv2
import numpy as np
import ollama
import config
from io_tools import print_progress, draw_boxes
from utils import get_images


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

    args = parser.parse_args()
    path_to_images = args.image_dir
    overwrite = args.overwrite
    images = get_images(path_to_images)
    images_count = len(images)
    time_per_image = []
    expected_schema = {cls: [["xmin", "ymin", "xmax", "ymax"]] for cls in config.CLASSES}
    log_file = os.path.join(config.ROOT_DIR, "errors.log")

    with open(log_file, "w") as _:
        pass

    detection_prompt = f"""
    Analyze the image and detect all instances of the following object classes: {', '.join(config.CLASSES)}.
    Return the results STRICTLY as a valid JSON object. The keys must be the exact class names provided, and the values must be lists of bounding boxes.
    Each bounding box must be represented as a list of four coordinates in the format [xmin, ymin, xmax, ymax].
    CRITICAL: Use the format [xmin, ymin, xmax, ymax].
    DO NOT use [ymin, xmin]. Double check that the first value is X and the second is Y.
    If a specific class is not found in the image, its value must be an empty list [].
    Do not include any conversational text, explanations, or markdown code blocks. Start directly with {{ and end with }}.
    Expected JSON structure:
    {expected_schema}
    """

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

        # --- Reading and encoding of image ---

        img = cv2.imread(image_path)
        success, buffer = cv2.imencode('.jpg', img)
        h, w = img.shape[:2]

        if not success:
            print(f"Failed to encode image {image_path}")
            print("Skipping annotation")
            continue

        img_bytes = buffer.tobytes()

        # --- Generation of the raw annotation ----

        if config.VERBOSE:
            print(f"Annotating image {os.path.basename(image_path)} {img_num + 1}/{images_count} [{(img_num + 1) / images_count * 100:0.2f} %]")

        t1 = time()

        response = ollama.generate(
            model=config.VLM_NAME,
            prompt=detection_prompt,
            images=[img_bytes],
            stream=False,
            options={'temperature': 0}
        )

        if config.VERBOSE:
            print(f"  * annotation done in {time() - t1:0.2f}s")

        # --- Refining the raw JSON with LLM ---

        t2 = time()
        raw_json = response["response"]

        correction_prompt = f"""
        Check and, if necessary, correct the JSON so that it matches the given structure. 
        The JSON structure must be: {{str: [[int, int, int, int], ...], ...}}.
        The classes names: {config.CLASSES}.
        Return ONLY the JSON object. Start directly with {{ and end with }}. No conversational text.
        JSON: {raw_json}
        """

        response = ollama.generate(
            model=config.LLM_NAME,
            prompt=correction_prompt,
            stream=False,
            format="json",
            options={'temperature': 0, 'num_ctx': 4096}
        )

        time_per_image.append(time() - t0)
        eta = np.mean(time_per_image) * (images_count - img_num - 1)
        eta_str = str(datetime.timedelta(seconds=round(eta)))

        if config.VERBOSE:
            print(f"  * refining done in {time() - t2:0.2f}s")
            print(f"  * total time for image: {time() - t1:0.2f}s")
            print(f"  * ETA: {eta_str}")

        try:
            boxes = json.loads(response["response"])
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"ERROR when annotating {image_path}\n")
                f.write(f"E: {e}\n")
                f.write(f"Response: {response['response']}\n")

            continue

        for cls, boxes_list in boxes.items():
            if len(boxes_list) > 0:
                if not isinstance(boxes_list[0], list):
                    boxes[cls] = [boxes_list]

        # --- Final JSON cleanup ---

        parsing_errors = False

        for cls, boxes_list in boxes.items():
            if len(boxes_list) > 0:
                for i, box in enumerate(boxes_list):
                    try:
                        boxes_list[i] = [int(b) for b in box]
                    except Exception as e:
                        with open(log_file, "a") as f:
                            f.write(f"ERROR when annotating {image_path}\n")
                            f.write(f"E: {e}\n")
                            f.write(f"Box: {box}\n")
                        parsing_errors = True
        if parsing_errors:
            continue

        # --- Saving annotation in YOLOv4 txt format ---

        lines = []
        if config.SCALE is not None:
            sx = sy = config.SCALE
        else:
            sx = w
            sy = h

        for cls_id, cls in enumerate(config.CLASSES):
            if cls not in boxes:
                continue

            for box in boxes[cls]:
                try:
                    x1, y1, x2, y2 = box[0] / sx, box[1] / sy, box[2] / sx, box[3] / sy
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
                draw_boxes(img, boxes, sx, sy)
                cv2.imshow("annotation", img)
                cv2.waitKey(1)
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"ERROR when annotating {image_path}\n")
                f.write(f"E: {e}\n")
                f.write(f"Boxes: {boxes}\n")

if __name__ == "__main__":
    main()