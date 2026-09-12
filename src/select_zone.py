import cv2
import json


zone_points = []
zone_saved = False


def mouse_callback(event, x, y, flags, param):

    global zone_points

    #Left click = thêm điểm
    if event == cv2.EVENT_LBUTTONDOWN:

        zone_points.append((x, y))

    #Right click = xóa điểm cuối
    elif event == cv2.EVENT_RBUTTONDOWN:

        if len(zone_points) > 0:

            zone_points.pop()


def draw_zone(frame):

    #Vẽ các điểm
    for point in zone_points:

        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    #Vẽ các đường nối
    if len(zone_points) >= 2:

        for i in range(len(zone_points) - 1):

            cv2.line(frame, zone_points[i], zone_points[i + 1], (0, 0, 255), 2)

    #Đóng polygon
    if len(zone_points) >= 3:

        cv2.line(frame, zone_points[-1], zone_points[0], (0, 0, 255), 2)

    return frame


def save_zone():

    if len(zone_points) < 3:

        print("Need at least 3 points to save a zone.")

        return False

    data = {
        "zone": zone_points
    }

    with open("zone.json", "w") as file:

        json.dump(data, file, indent=4)

    print("Zone saved to zone.json.")

    return True


def draw_instructions(frame):

    instructions = [
        "LEFT CLICK  : Add point",
        "RIGHT CLICK : Remove last point",
        "C            : Clear all points",
        "S            : Save zone",
        "Q            : Quit"
    ]

    x = 20
    y = 30

    for text in instructions:

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y += 30

    return frame


def select_zone_mode(video_path):

    global zone_points, zone_saved

    #Reset zone mỗi lần mở editor
    zone_points = []
    zone_saved = False

    video = cv2.VideoCapture(video_path)

    ret, frame = video.read()

    if not ret:

        print("Cannot read video.")

        video.release()

        return

    cv2.namedWindow("Select Zone")

    cv2.setMouseCallback("Select Zone", mouse_callback)

    while True:

        display_frame = frame.copy()

        display_frame = draw_zone(display_frame)

        display_frame = draw_instructions(display_frame)

        cv2.imshow("Select Zone", display_frame)

        key = cv2.waitKey(1) & 0xFF

        #C = clear toàn bộ zone
        if key == ord("c"):

            zone_points.clear()

            print("Zone cleared.")

        #S = save zone
        elif key == ord("s"):

            if save_zone():

                zone_saved = True

                print("Zone saved.")

        #Q = quit
        elif key == ord("q"):

            break

    video.release()

    cv2.destroyWindow("Select Zone")


if __name__ == "__main__":

    select_zone_mode("videos/input.mp4")