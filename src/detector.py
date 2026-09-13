import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

people_state = {}

#Số frame liên tục cần thấy 1 state mới trước khi tin là thật
#(chống nhiễu khi center point dao động sát viền zone)
#Giá trị mặc định ở đây chỉ để phòng hờ, thực tế sẽ được
#set_confirm_time() tính lại theo FPS thật của video khi chạy main.py
CONFIRM_FRAMES = 3


def set_confirm_time(fps, seconds=0.5):

    global CONFIRM_FRAMES

    #Quy đổi thời gian (giây) sang số frame dựa theo FPS thật của video
    CONFIRM_FRAMES = max(1, round(fps * seconds))

    print(
        f"CONFIRM_FRAMES = {CONFIRM_FRAMES} "
        f"(~{seconds}s ở {fps:.2f} FPS)"
    )


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
            "tracking": "ACTIVE",
            "pending": None,
            "pending_count": 0
        }

        return people_state[track_id]

    #ID đã tồn tại
    state = people_state[track_id]

    state["tracking"] = "ACTIVE"

    #Detect ra giống với current đang giữ -> không có gì để confirm,
    #hủy pending nếu có (tránh trường hợp dao động qua lại rồi lại về cũ)
    if current_state == state["current"]:

        state["pending"] = None
        state["pending_count"] = 0

        return state

    #Detect ra khác với current -> chưa vội tin, chờ xác nhận đủ N frame liên tục
    if state["pending"] == current_state:

        state["pending_count"] += 1

    else:

        state["pending"] = current_state
        state["pending_count"] = 1

    #Chưa đủ N frame liên tục -> coi là nhiễu, giữ nguyên current/entered
    if state["pending_count"] < CONFIRM_FRAMES:

        return state

    #Đã đủ N frame liên tục -> confirm transition thật sự
    previous_state = state["current"]

    state["previous"] = previous_state
    state["current"] = current_state

    #OUTSIDE -> ZONE = người đi vào nhà
    if previous_state == "OUTSIDE" and current_state == "ZONE":

        state["entered"] = True

    #ZONE -> OUTSIDE = người đi ra khỏi nhà
    elif previous_state == "ZONE" and current_state == "OUTSIDE":

        state["entered"] = False

    #ZONE -> ZONE hoặc OUTSIDE -> OUTSIDE = giữ nguyên entered
    #Trường hợp reappear sau LOST cũng rơi vào 1 trong các nhánh trên,
    #vì previous_state ở đây là "current" cũ được giữ nguyên lúc LOST

    state["pending"] = None
    state["pending_count"] = 0

    return state