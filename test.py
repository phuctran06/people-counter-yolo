import cv2

video = cv2.VideoCapture("videos/input.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter( "videos/test.mp4", fourcc, fps, (width, height) )

max_frames = int(fps * 15)

for _ in range(max_frames):
    ret, frame = video.read()

    if not ret:
        break

    output.write(frame)

video.release()
output.release()