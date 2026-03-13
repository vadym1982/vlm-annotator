import sys
import cv2

cls_colors = [
    (127, 0, 255),
    (0, 127, 255),
    (127, 255, 0),
    (255, 0, 127),
    (0, 255, 127),
    (255, 127, 0)
]


def print_progress(num, count, eta, bar_length=50):
    """
    Displays a dynamic ASCII progress bar in the terminal.

    This function calculates the completion percentage based on the current
    iteration and total count, then prints a visual bar with an estimated
    time of arrival (ETA).

    Args:
        num (int): The current progress value (current iteration).
        count (int): The total number of iterations to complete.
        eta (str): Formatted string representing the estimated time remaining.
        bar_length (int, optional): The total character width of the progress bar.
            Defaults to 50.

    Example:
         50.00 % [=========================>..........] ETA: 0:01:24
    """
    percent = (num / count) * 100
    arrow_length = int(round(percent / 100 * bar_length))

    if arrow_length > 0:
        bar = '=' * (arrow_length - 1) + '>'
    else:
        bar = ''

    dots = '.' * (bar_length - len(bar))
    sys.stdout.write(f"\r{percent:6.2f} % [{bar}{dots}] ETA: {eta}")
    sys.stdout.flush()


def draw_boxes(img, boxes, sx, sy):
    """
    Draws colored bounding boxes on the image with coordinate re-scaling.

    This function iterates through detected objects, scales their coordinates
    from a model-specific space (defined by sx/sy) to the actual image
    pixel dimensions, and renders them using OpenCV.

    Args:
        img (numpy.ndarray): The image array (BGR) to draw on.
        boxes (dict): A dictionary where keys are class names (str) and
            values are lists of bounding boxes. Each box is [xmin, ymin, xmax, ymax].
        sx (float): The horizontal scaling factor used for normalization
            (e.g., 1000 for Qwen-VL or image width).
        sy (float): The vertical scaling factor used for normalization
            (e.g., 1000 for Qwen-VL or image height).

    Note:
        The color for each class is automatically selected from the global
        `cls_colors` list using a modulo operator.
    """
    h, w = img.shape[:2]

    for i, (cls, boxes) in enumerate(boxes.items()):
        color_num = i % len(cls_colors)

        for box in boxes:
            x1 = round(box[0] / sx * w)
            y1 = round(box[1] / sy * h)
            x2 = round(box[2] / sx * w)
            y2 = round(box[3] / sy * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), cls_colors[color_num], 2)
