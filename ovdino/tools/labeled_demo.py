import os
import cv2
import json
import random
from detectron2.structures import BoxMode

def read_label_file(label_path):
    with open(label_path, 'r') as file:
        line = file.readline().strip()
        parts = line.split()
        class_label = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    return class_label, x_center, y_center, width, height

def renormalize_bbox(class_label, x_center, y_center, width, height, image_width, image_height):
    x_center = x_center * image_width
    y_center = y_center * image_height
    width = width * image_width
    height = height * image_height
    
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)
    
    return class_label, x_min, y_min, x_max, y_max

def resize_bbox(x_min, y_min, x_max, y_max, original_width, original_height, new_width, new_height):
    x_min = x_min * new_width / original_width
    y_min = y_min * new_height / original_height
    x_max = x_max * new_width / original_width
    y_max = y_max * new_height / original_height
    return x_min, y_min, x_max, y_max

def annotate_image_with_detectron2(image_path, label_path):
    """
    Annotates an image using the label from the corresponding text file.

    Args:
        image_path (str): Path to the input image.
        label_path (str): Path to the label file.

    Returns:
        dict: Annotation data ready for Detectron2 in JSON format.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load the image {image_path}.")
        return

    # Read the label
    class_label, x_center, y_center, width, height = read_label_file(label_path)
    original_height, original_width = image.shape[:2]
    class_label, x_min, y_min, x_max, y_max = renormalize_bbox(class_label, x_center, y_center, width, height, original_width, original_height)
    
    # Resize the image to 640x640
    new_width, new_height = 640, 640
    image = cv2.resize(image, (new_width, new_height))
    
    # Adjust the bounding box coordinates
    x_min, y_min, x_max, y_max = resize_bbox(x_min, y_min, x_max, y_max, original_width, original_height, new_width, new_height)
    #show annotated image
    # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    # cv2.imshow(f'{class_label}', image)
    # cv2.waitKey(500)
    
    # Create the annotation
    annotation = {
        "iscrowd": 0,
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "category_id": class_label,
        "segmentation": [],
        "bbox_mode": BoxMode.XYWH_ABS,
        "filename": os.path.basename(image_path),
    }

    return annotation

def annotate_folder(image_folder, output_file):
    """
    Annotates all images in a folder using their corresponding label files.

    Args:
        image_folder (str): Path to the folder containing images and label files.
        output_file (str): Path to the output JSON file to save annotations.
    """
    annotations = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join("datasets/DatasetToBeEval/Support/labels", os.path.splitext(filename)[0] + ".txt")
            if os.path.exists(label_path):
                annotation = annotate_image_with_detectron2(image_path, label_path)
                if annotation:
                    annotations.append(annotation)
            else:
                print(f"Label file {label_path} not found.")

    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f"Annotations saved to {output_file}.")

# Example usage
if __name__ == "__main__":
    annotate_folder("datasets/DatasetToBeEval/Support/images", "support_annotations.json")