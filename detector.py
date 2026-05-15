import json
import cv2
import numpy as np
import ollama
import config
from io_tools import draw_boxes

expected_schema = {cls: [["xmin", "ymin", "xmax", "ymax"]] for cls in config.CLASSES}

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

def detect(img):
    h, w = img.shape[:2]
    mp = w * h / 1e6

    if mp > config.MP_LIMIT:
        factor = np.sqrt(config.MP_LIMIT / mp)
        inp = cv2.resize(img, (int(w * factor), int(h * factor)))
    else:
        inp = img

    success, buffer = cv2.imencode('.jpg', inp)

    if not success:
        return None, ["Failed to encode image"], ""

    img_bytes = buffer.tobytes()

    # --- Detection stage ---

    response = ollama.generate(
        model=config.VLM_NAME,
        prompt=detection_prompt,
        images=[img_bytes],
        stream=False,
        options={'temperature': 0}
    )

    raw_json = response["response"]

    # --- Refining output ---

    correction_prompt = f"""
    Check and, if necessary, correct the JSON so that it matches the given structure. 
    The JSON structure must be: {{str: [[int, int, int, int], ...], ...}}.
    The classes names: {config.CLASSES}.
    If a specific class is not found in the image, its value must be an empty list [].
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

    try:
        boxes = json.loads(response["response"])
    except Exception as e:
        return None, str(e), response["response"]

    for cls, boxes_list in boxes.items():
        if len(boxes_list) > 0:
            if not isinstance(boxes_list[0], list):
                boxes[cls] = [boxes_list]

    # --- Final JSON cleanup ---

    parsing_errors = []

    if config.SCALE is not None:
        sx = sy = config.SCALE
    else:
        sx = w
        sy = h

    for cls, boxes_list in boxes.items():
        if len(boxes_list) > 0:
            for i, box in enumerate(boxes_list):
                try:
                    boxes_list[i] = [int(box[0]) / sx, int(box[1]) / sy, int(box[2]) / sx, int(box[3]) / sy]
                except Exception as e:
                    parsing_errors.append(str(e))
    if len(parsing_errors) > 0:
        return None, parsing_errors, response["response"]

    return boxes, [], response["response"]


def calculate_slice_coords(length, num_tiles, overlap_ratio):
    """
    Calculates start and end coordinates for slices along a single axis.
    Ensures the last slice ends exactly at the image edge.
    """
    if num_tiles <= 1:
        return [(0, length)]

    # Slice size calculation formula:
    # L = num_tiles * S - (num_tiles - 1) * (S * overlap)
    slice_size = int(length / (num_tiles - (num_tiles - 1) * overlap_ratio))
    step = int(slice_size * (1 - overlap_ratio))

    slices = []
    for i in range(num_tiles):
        start = i * step
        end = start + slice_size

        # For the last slice, hard-bind the end to the image edge,
        # shifting the start back to preserve the slice size (important for the model)
        if i == num_tiles - 1:
            end = length
            start = max(0, end - slice_size)

        slices.append((start, end))

    return slices


def non_max_suppression(boxes, iou_threshold):
    """
    Classic NMS (Non-Maximum Suppression) algorithm using NumPy.
    boxes: list of lists [xmin, ymin, xmax, ymax, score]
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # Calculate the area of each box
    areas = (x2 - x1) * (y2 - y1)

    # Sort box indices by score in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # Take the box with the highest score
        i = order[0]
        keep.append(boxes[i].tolist())

        # Calculate overlap coordinates with all other boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Calculate overlap width and height
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Calculate Intersection over Union (IoU)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep only boxes where IoU is less than the specified threshold
        inds = np.where(iou <= iou_threshold)[0]
        # +1 because order[1:] shifts indices by 1
        order = order[inds + 1]

    return keep


def detect_sahi(image, rows, cols, overlap_w, overlap_h, nms_threshold=0.5):
    """
    Slices the image, runs detection on each tile, and merges the results.
    """
    height, width = image.shape[:2]

    # Calculate slice coordinates along both axes
    x_slices = calculate_slice_coords(width, cols, overlap_w)
    y_slices = calculate_slice_coords(height, rows, overlap_h)

    global_detections = {}

    # Iterate through each tile
    for row, (y_start, y_end) in enumerate(y_slices):
        for col, (x_start, x_end) in enumerate(x_slices):
            # Extract the tile (patch)
            patch = image[y_start:y_end, x_start:x_end]
            patch_height, patch_width = patch.shape[:2]

            # Invoke the detector
            # Expected format: {'class_name': [[xmin, ymin, xmax, ymax, score], ...]}
            patch_detections, errors, response = detect(patch)

            if config.VERBOSE:
                print(f"Patch: {row + 1}[{rows}], {col}[{cols}]. Detections: {patch_detections}")

            if len(errors) > 0:
                return None, errors, patch_detections, response

            # Shift coordinates and add to the global list
            for class_name, bboxes in patch_detections.items():
                if class_name not in global_detections:
                    global_detections[class_name] = []

                for bbox in bboxes:
                    # If score is missing, add 1.0 as a placeholder
                    xmin, ymin = bbox[0] * patch_width, bbox[1] * patch_height
                    xmax, ymax = bbox[2] * patch_width, bbox[3] * patch_height
                    score = bbox[4] if len(bbox) == 5 else 1.0

                    # Coordinate shift
                    global_xmin = xmin + x_start
                    global_ymin = ymin + y_start
                    global_xmax = xmax + x_start
                    global_ymax = ymax + y_start

                    # Clipping to ensure coordinates stay within original image boundaries
                    global_xmin = max(0, min(global_xmin, width))
                    global_ymin = max(0, min(global_ymin, height))
                    global_xmax = max(0, min(global_xmax, width))
                    global_ymax = max(0, min(global_ymax, height))

                    global_detections[class_name].append(
                        [global_xmin, global_ymin, global_xmax, global_ymax, score]
                    )

    # Apply NMS for each class separately to remove duplicates at tile boundaries
    final_results = {}
    for class_name, bboxes in global_detections.items():
        final_results[class_name] = non_max_suppression(bboxes, iou_threshold=nms_threshold)

    # Normalize coordinates
    scales = [width, height, width, height]

    for class_name in final_results.keys():
        for i in range(len(final_results[class_name])):
            final_results[class_name][i] = [final_results[class_name][i][j] / scales[j] for j in range(4)]

    return final_results, [], ""
