import cv2


def read_video(video_path):
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break

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

def draw_people(frame, people):
    for person in people:
        x1, y1, x2, y2 = person

        cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)

    return frame

video = cv2.VideoCapture("videos/input.mp4")

get_video_info(video)

video.release()