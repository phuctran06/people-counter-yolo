import cv2
import json
import numpy as np

from detector import track_people, get_person_state, people_state


#Số frame cho phép một ID biến mất tạm thời
MAX_MISSING_FRAMES = 10


def read_video(video_path, zone):

    video = cv2.VideoCapture(video_path)

    previous_ids = []

    missing_frames = {}

    entered_count = 0

    while True:

        ret, frame = video.read()

        if not ret:
            break

        people = track_people(frame)

        current_ids = []

        person_states = {}

        for person in people:

            track_id = person[5]

            current_ids.append(track_id)

            #ID xuất hiện lại thì reset missing frames
            missing_frames[track_id] = 0

            state = get_person_state(person, zone)

            person_states[track_id] = state

            previous_state = state["previous"]
            current_state = state["current"]

            #OUTSIDE -> ZONE = người đi vào nhà
            if previous_state == "OUTSIDE" and current_state == "ZONE":

                if not state["entered"]:

                    entered_count += 1

                    state["entered"] = True

                    print(f"Person {track_id}: ENTERED")

            #ZONE -> OUTSIDE = người đi ra khỏi nhà
            elif previous_state == "ZONE" and current_state == "OUTSIDE":

                if state["entered"]:

                    entered_count -= 1

                    state["entered"] = False

                    print(f"Person {track_id}: LEFT")

            print(
                f"ID {track_id}: "
                f"{current_state}, "
                f"Entered: {state['entered']}"
            )

        #Số người đang được tracking trên frame hiện tại
        people_count = len(current_ids)

        #Tìm ID vừa biến mất
        disappeared_ids = find_lost_people(previous_ids, current_ids)

        #Tăng số frame bị mất
        for track_id in disappeared_ids:

            if track_id not in missing_frames:

                missing_frames[track_id] = 0

            missing_frames[track_id] += 1

        #Kiểm tra ID đã LOST thật sự
        lost_ids = []

        for track_id in list(missing_frames.keys()):

            if missing_frames[track_id] >= MAX_MISSING_FRAMES:

                lost_ids.append(track_id)

                del missing_frames[track_id]

        #LOST không làm thay đổi Entered
        for track_id in lost_ids:

            if track_id not in people_state:

                continue

            state = people_state[track_id]

            state["tracking"] = "LOST"

            print(
                f"Person {track_id}: LOST | "
                f"Entered: {state['entered']}"
            )

        #Hiển thị thông tin
        print("Current IDs:", current_ids)
        print("People:", people_count)
        print("Missing frames:", missing_frames)
        print("Lost IDs:", lost_ids)
        print("Entered:", entered_count)

        frame = draw_people(
            frame,
            people,
            zone,
            people_count,
            entered_count,
            person_states
        )

        cv2.imshow("People Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

        previous_ids = current_ids

    video.release()

    cv2.destroyAllWindows()


def get_video_info(video):

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps

    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")


def load_zone():

    with open("zone.json", "r") as file:

        data = json.load(file)

    zone = np.array(data["zone"], dtype=np.int32)

    return zone


def draw_people(frame, people, zone, people_count, entered_count, person_states):

    #Vẽ zone
    cv2.polylines(
        frame,
        [zone],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2
    )

    for person in people:

        x1, y1, x2, y2, confidence, track_id = person

        state = person_states[track_id]

        current_state = state["current"]

        #Tính center point
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        #ZONE = cam, OUTSIDE = xanh lá
        if current_state == "ZONE":

            text_color = (0, 165, 255)

        else:

            text_color = (0, 255, 0)

        #Vẽ bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            text_color,
            2
        )

        #Hiển thị ID, confidence và current state
        cv2.putText(
            frame,
            f"ID: {track_id} Person: {confidence:.2f}, Current: {current_state}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            15
        )

        #Vẽ center point
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

    #Hiển thị số người đang được tracking
    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    #Hiển thị số người đang ở trong nhà
    cv2.putText(
        frame,
        f"Entered: {entered_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    return frame


def is_inside_zone(person, zone):

    if zone is None or len(zone) < 3:

        return False

    x1, y1, x2, y2, _ = person

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    return cv2.pointPolygonTest(
        zone,
        (center_x, center_y),
        False
    ) >= 0


def find_lost_people(previous_ids, current_ids):

    lost_ids = []

    for track_id in previous_ids:

        if track_id not in current_ids:

            lost_ids.append(track_id)

    return lost_ids


if __name__ == "__main__":

    video = cv2.VideoCapture("videos/input.mp4")

    zone = load_zone()

    read_video("videos/input.mp4", zone)

    get_video_info(video)

    video.release()