import cv2

from detector import detect_people


def read_video(video_path):
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        people = detect_people(frame)

        frame = draw_people(frame, people)

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


def draw_people(frame, people, zone):

    cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)

    for person in people:
        x1, y1, x2, y2, confidence = person

        # Calculate the center of the bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)

        cv2.putText(frame,f"Person: {confidence:.2f}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),20 )
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    return frame


def is_inside_zone(person, zone):
    x1, y1, x2, y2, _ = person
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    

    return cv2.pointPolygonTest(zone, (center_x, center_y), False) >= 0




if __name__ == "__main__":

    video = cv2.VideoCapture("videos/input.mp4")

    read_video("videos/input.mp4")
    get_video_info(video)

    video.release()
