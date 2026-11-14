from ultralytics import YOLO
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
import json

# 1. Load your trained model
model = YOLO("Vision/models/best.pt")
# 2. Run detection on test images
results = model.predict(
    source="dataset/test/images",
    imgsz=640,
    conf=0.25,
    save=True,
    project="runs/detect",
    name="coordinate_test",
    exist_ok=True
)

all_detections = []


# 3. Extract coordinates from every detected object
for result in results:

    if not isinstance(result, Results):
        raise TypeError(f"Expected an Ultralytics Results object, got {type(result).__name__}")

    image_detections = []

    if result.boxes is not None:

        for box in result.boxes:

            # Class and colour
            class_id = int(box.cls[0].item())
            color = model.names[class_id]

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Calculate centre, width and height
            width = x2 - x1
            height = y2 - y1

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            confidence = float(box.conf[0].item())

            detection = {
                "color": color,
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "center_x": round(center_x, 1),
                "center_y": round(center_y, 1),
                "width": round(width, 1),
                "height": round(height, 1)
            }

            image_detections.append(detection)

    all_detections.append({
        "image": Path(result.path).name,
        "objects": image_detections
    })

    print(f"\nImage: {Path(result.path).name}")

    for obj in image_detections:
        print(obj)

output_path = Path("runs/detect/coordinate_test/detections.json")

# Create the folder if it doesn't exist
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as file:
    json.dump(all_detections, file, indent=4)

print(f"\nCoordinates saved to: {output_path}")