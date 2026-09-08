import cv2
import json
import numpy as np

from detector import track_people



def read_video(video_path, zone):

    video = cv2.VideoCapture(video_path)

    previous_ids = []

    while True:

        ret, frame = video.read()
        if not ret:
            break

        people = track_people(frame)

        current_ids = []

        for person in people:
            track_id = person[5]
            current_ids.append(track_id)

        lost_ids = find_lost_people(previous_ids, current_ids)

        print("Current IDs:", current_ids)
        print("Lost IDs:", lost_ids)

        frame = draw_people(frame, people, zone)

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



def draw_people(frame, people, zone):

    cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)

    for person in people:
        x1, y1, x2, y2, confidence, track_id = person

        #Calculate the center of the bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} Person: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 20)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame


def is_inside_zone(person, zone):

    if zone is None or len(zone) < 3:
        return False
    
    x1, y1, x2, y2, _ = person
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    return cv2.pointPolygonTest(zone, (center_x, center_y), False) >= 0


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

