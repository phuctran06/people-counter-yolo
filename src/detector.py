import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

people_state = {}


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

def track_people(frame):

    results = model.track(frame, persist=True)

    people = []

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if box.id is not None:
                    track_id = int(box.id[0])

                    people.append((x1, y1, x2, y2, confidence, track_id))

    return people


def get_person_state(person, zone):

    x1, y1, x2, y2, _, track_id = person

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    inside_zone = cv2.pointPolygonTest(zone, (center_x, center_y), False) >= 0

    if inside_zone:
        current_state = "ZONE"
    else:
        current_state = "OUTSIDE"

    if track_id not in people_state:
        people_state[track_id] = {
            "previous": current_state,
            "current": current_state
        }

    else:
        people_state[track_id]["previous"] = people_state[track_id]["current"]
        people_state[track_id]["current"] = current_state


    return people_state[track_id]