import cv2
import json
import random
from detectron2.structures import BoxMode
def annotate_image_with_detectron2(image_path):
    """
    Allows user to crop an image, input category ID, and outputs the annotation data
    in a format compatible with Detectron2.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Annotation data ready for Detectron2 in JSON format.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image.")
        return

    # Initialize variables
    bbox = []  # Current bounding box being cropped
    cropping = False
    annotations = []  # Store all annotations
    category_colors = {}  # Map categories to colors

    def get_color_for_category(category_id):
        """Assign a consistent color for a category."""
        if category_id not in category_colors:
            category_colors[category_id] = [random.randint(0, 255) for _ in range(3)]
        return category_colors[category_id]

    def mouse_callback(event, x, y, flags, param):
        nonlocal cropping, bbox
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start cropping
            cropping = True
            bbox = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and cropping:
            # Update width and height as the mouse moves
            bbox[2] = x - bbox[0]
            bbox[3] = y - bbox[1]
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish cropping
            cropping = False
            bbox[2] = x - bbox[0]
            bbox[3] = y - bbox[1]

    # Create a named window and set the callback
    cv2.namedWindow("Crop Image")
    cv2.setMouseCallback("Crop Image", mouse_callback)

    while True:
        # Display the image with the bounding boxes
        clone = image.copy()
        # Draw all saved annotations
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            color = get_color_for_category(ann["category_id"])
            cv2.rectangle(clone, (x, y), (x + w, y + h), color, 2)
            cv2.putText(clone, f"ID: {ann['category_id']}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw the current cropping box (if defined)
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Crop Image", clone)

        # Key handling
        key = cv2.waitKey(1)
        if key == ord("q"):  # Quit and save
            print("Annotation session ended. Saving annotations...")
            break
        elif key == ord("s"):  # Save the current box
            if bbox:
                # Ask user for the category ID
                category_id = input("Enter the category ID for the selected region: ")
                # Append the annotation
                annotations.append({
                    "iscrowd": 0,
                    "bbox": bbox.copy(),  # Store a copy of the bounding box
                    "category_id": int(category_id),
                    "segmentation": [],  # Can be filled later for segmentation
                    "bbox_mode": BoxMode.XYWH_ABS,  # Detectron2's box mode
                })
                print(f"Saved box {bbox} with category ID {category_id}.")
                bbox = []  # Reset current bounding box

    cv2.destroyAllWindows()
    print("Annotation data:")
    print(annotations)
    return annotations


# Example usage
if __name__ == "__main__":
    annotate_image_with_detectron2("demo/imgs/000000001584.jpg")
