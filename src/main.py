import cv2
import json
import numpy as np

from detector import track_people, get_person_state, people_state


#Số frame cho phép một ID biến mất tạm thời
MAX_MISSING_FRAMES = 10


def read_video(video_path, zone):

    video = cv2.VideoCapture(video_path)

    missing_frames = {}

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

            #ID xuất hiện ở frame này -> reset missing frames
            missing_frames[track_id] = 0

            #Lưu lại entered trước khi update, chỉ để in log ENTERED/LEFT
            previous_entered = people_state.get(track_id, {}).get("entered", False)

            state = get_person_state(person, zone)

            person_states[track_id] = state

            if state["entered"] and not previous_entered:

                print(f"Person {track_id}: ENTERED")

            elif not state["entered"] and previous_entered:

                print(f"Person {track_id}: LEFT")

            print(
                f"ID {track_id}: "
                f"{state['current']}, "
                f"Entered: {state['entered']}"
            )

        #Tăng missing_frames cho MỌI ID đã biết mà không xuất hiện ở frame này
        #(so với current_ids, không so với frame trước, nên đếm đúng số frame
        #liên tiếp bị mất thay vì chỉ tăng đúng 1 lần)
        for track_id in list(people_state.keys()):

            if track_id in current_ids:

                continue

            missing_frames[track_id] = missing_frames.get(track_id, 0) + 1

            if (
                missing_frames[track_id] >= MAX_MISSING_FRAMES
                and people_state[track_id]["tracking"] != "LOST"
            ):

                people_state[track_id]["tracking"] = "LOST"

                #LOST không làm thay đổi entered
                print(
                    f"Person {track_id}: LOST | "
                    f"Entered: {people_state[track_id]['entered']}"
                )

        #Số người đang được tracking trên frame hiện tại
        people_count = len(current_ids)

        #Entered luôn được tính lại từ people_state, không cộng/trừ tay
        #=> tự động đúng cả với ID mới đã nằm sẵn trong ZONE (Case 9)
        entered_count = sum(
            1 for state in people_state.values() if state["entered"]
        )

        lost_ids = [
            track_id for track_id, state in people_state.items()
            if state["tracking"] == "LOST"
        ]

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
            2
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


if __name__ == "__main__":

    video = cv2.VideoCapture("videos/input.mp4")

    zone = load_zone()

    read_video("videos/input.mp4", zone)

    get_video_info(video)

    video.release()