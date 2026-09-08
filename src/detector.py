from ultralytics import YOLO


model = YOLO("yolo11n.pt")

def detect_people(frame):
    results = model(frame)

    people = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  

            if class_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                people.append((x1, y1, x2, y2, confidence))

    return people