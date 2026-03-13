import os
import json
import argparse
from PIL import Image
from utils import get_images  # Using your provided utility


def get_classes(dataset_dir):
    """
    Attempts to read classes.txt in the dataset directory.
    Returns a list of class names or an empty list if not found.
    """
    classes_file = os.path.join(dataset_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []


def parse_yolo_line(line, img_w, img_h):
    """
    Converts YOLO normalized coordinates to absolute pixel coordinates.
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(parts[0])
    x_c = float(parts[1]) * img_w
    y_c = float(parts[2]) * img_h
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h

    x_min = int(round(x_c - (w / 2)))
    y_min = int(round(y_c - (h / 2)))
    width = int(round(w))
    height = int(round(h))

    return class_id, x_min, y_min, width, height


def convert_to_coco(image_paths, classes):
    """
    Converts YOLO annotations to COCO JSON format.
    """
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    class_mapping = {i: i + 1 for i in range(len(classes))}
    for i, cls_name in enumerate(classes):
        coco_dict["categories"].append({"id": i + 1, "name": cls_name})

    annotation_id = 1

    for img_id, img_path in enumerate(image_paths):
        img_filename = os.path.basename(img_path)

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        coco_dict["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": img_w,
            "height": img_h
        })

        txt_path = os.path.splitext(img_path)[0] + '.txt'

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    bbox_data = parse_yolo_line(line, img_w, img_h)
                    if not bbox_data:
                        continue

                    yolo_class_id, x, y, w, h = bbox_data

                    if yolo_class_id not in class_mapping:
                        cat_id = len(class_mapping) + 1
                        coco_dict["categories"].append({"id": cat_id, "name": f"class_{yolo_class_id}"})
                        class_mapping[yolo_class_id] = cat_id

                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_mapping[yolo_class_id],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    annotation_id += 1

    return coco_dict


def convert_to_via(image_paths, classes):
    """
    Converts YOLO annotations to VGG Image Annotator (VIA) JSON format.
    """
    via_dict = {}

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        file_size = os.path.getsize(img_path)

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # VIA key is filename + filesize (matches your results_json.json example)
        via_key = f"{img_filename}{file_size}"

        regions = []
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    bbox_data = parse_yolo_line(line, img_w, img_h)
                    if not bbox_data:
                        continue

                    class_id, x, y, w, h = bbox_data
                    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

                    regions.append({
                        "shape_attributes": {
                            "name": "rect",
                            "x": x, "y": y,
                            "width": w, "height": h
                        },
                        "region_attributes": {
                            "class": class_name
                        }
                    })

        via_dict[via_key] = {
            "filename": img_filename,
            "size": file_size,
            "regions": regions,
            "file_attributes": {}
        }

    return via_dict


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO or VIA format.")
    parser.add_argument("dataset_dir", type=str, help="Directory containing images and .txt files")
    parser.add_argument("--format", type=str, choices=['coco', 'via'], required=True, help="Target format")

    args = parser.parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)

    if not os.path.isdir(dataset_dir):
        print(f"Error: {dataset_dir} not found.")
        return

    # Use get_images from utils.py (returns full paths)
    image_paths = get_images(dataset_dir)

    if not image_paths:
        print(f"No valid images found in {dataset_dir}.")
        return

    print(f"Processing {len(image_paths)} images for {args.format.upper()} format...")
    classes = get_classes(dataset_dir)

    if args.format == 'coco':
        output_data = convert_to_coco(image_paths, classes)
    else:
        output_data = convert_to_via(image_paths, classes)

    # Output file: next to the dataset folder, named after the folder
    parent_dir = os.path.dirname(dataset_dir)
    dataset_name = os.path.basename(dataset_dir)
    output_path = os.path.join(parent_dir, f"{dataset_name}.json")

    with open(output_path, 'w') as out_f:
        json.dump(output_data, out_f, indent=2)

    print(f"Done. File saved to: {output_path}")


if __name__ == "__main__":
    main()