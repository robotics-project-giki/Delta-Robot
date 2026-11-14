
import json
import cv2
from pathlib import Path

# EXAMPLE workspace dimensions — replace these later
WORKSPACE_WIDTH_MM = 300
WORKSPACE_HEIGHT_MM = 200

# Folder containing test images
IMAGE_FOLDER = Path("dataset/test/images")

# YOLO detection results
JSON_FILE = Path("runs/detect/coordinate_test/detections.json")


def pixel_to_robot(pixel_x, pixel_y, image_width, image_height):

    # Convert pixel coordinates into millimetres
    x_mm = (pixel_x / image_width) * WORKSPACE_WIDTH_MM
    y_mm = (pixel_y / image_height) * WORKSPACE_HEIGHT_MM

    # Move the origin to the centre of the workspace
    robot_x = x_mm - (WORKSPACE_WIDTH_MM / 2)

    # Reverse Y because image coordinates increase downwards
    robot_y = (WORKSPACE_HEIGHT_MM / 2) - y_mm

    return round(robot_x, 2), round(robot_y, 2)


# Load YOLO detections
with open(JSON_FILE, "r") as file:
    detections = json.load(file)

for image_data in detections:

    image_path = IMAGE_FOLDER / image_data["image"]
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    height, width = image.shape[:2]

    print(f"\nImage: {image_data['image']}")

    for obj in image_data["objects"]:

        x, y = pixel_to_robot(
            obj["center_x"],
            obj["center_y"],
            width,
            height
        )

        print(
            f"{obj['color']} block | "
            f"Pixel: ({obj['center_x']}, {obj['center_y']}) | "
            f"Robot: X={x} mm, Y={y} mm"
        )
