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

                    people.append(
                        (x1, y1, x2, y2, confidence, track_id)
                    )

    return people


def get_current_zone_state(person, zone):

    x1, y1, x2, y2, _, track_id = person

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    inside_zone = cv2.pointPolygonTest(
        zone,
        (center_x, center_y),
        False
    ) >= 0

    if inside_zone:

        return "ZONE"

    else:

        return "OUTSIDE"


def get_person_state(person, zone):

    track_id = person[5]

    current_state = get_current_zone_state(person, zone)

    #ID lần đầu xuất hiện
    if track_id not in people_state:

        people_state[track_id] = {
            "previous": current_state,
            "current": current_state,
            #Nếu ID mới đã nằm sẵn trong ZONE thì tính luôn là entered
            "entered": current_state == "ZONE",
            "tracking": "ACTIVE"
        }

        return people_state[track_id]

    #ID đã tồn tại, cập nhật state theo transition
    state = people_state[track_id]

    previous_state = state["current"]

    state["previous"] = previous_state
    state["current"] = current_state
    state["tracking"] = "ACTIVE"

    #OUTSIDE -> ZONE = người đi vào nhà
    if previous_state == "OUTSIDE" and current_state == "ZONE":

        state["entered"] = True

    #ZONE -> OUTSIDE = người đi ra khỏi nhà
    elif previous_state == "ZONE" and current_state == "OUTSIDE":

        state["entered"] = False

    #ZONE -> ZONE hoặc OUTSIDE -> OUTSIDE = giữ nguyên entered
    #Trường hợp reappear sau LOST cũng rơi vào 1 trong các nhánh trên,
    #vì previous_state ở đây là "current" cũ được giữ nguyên lúc LOST

    return state