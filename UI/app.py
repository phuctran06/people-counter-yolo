import os
import sys
import json
import time
import cv2
import numpy as np
import streamlit as st


#File này nằm trong UI/, còn detector.py / select_zone.py nằm trong src/
#(src/ là thư mục con của thư mục gốc project, cùng cấp với UI/)
#-> phải tự thêm src/ vào sys.path thì mới import được,
#bất kể bạn chạy lệnh streamlit từ đâu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:

    sys.path.insert(0, SRC_DIR)

from detector import track_people, get_person_state, set_confirm_time, people_state
from select_zone import select_zone_mode


#Đường dẫn video và file zone dùng chung cho cả 2 chức năng
#Tính theo PROJECT_ROOT (không phải theo cwd) để chạy đúng dù bạn
#chạy lệnh streamlit từ thư mục nào
VIDEO_PATH = os.path.join(PROJECT_ROOT, "videos", "input.mp4")
ZONE_PATH = os.path.join(PROJECT_ROOT, "zone.json")

#Số frame cho phép 1 ID biến mất tạm thời trước khi coi là LOST thật
MAX_MISSING_FRAMES = 10

#Thời gian (giây) cần thấy 1 state mới liên tục thì mới tin là transition thật
CONFIRM_SECONDS = 0.5

#Chiều rộng hiển thị video trên web, để giảm băng thông/độ trễ
DISPLAY_WIDTH = 720


def zone_file_is_valid():

    if not os.path.exists(ZONE_PATH):

        return False

    try:

        with open(ZONE_PATH, "r") as file:

            data = json.load(file)

    except (json.JSONDecodeError, OSError):

        return False

    zone_points = data.get("zone", [])

    #Cần ít nhất 3 điểm mới tạo được polygon hợp lệ
    return len(zone_points) >= 3


def load_zone():

    with open(ZONE_PATH, "r") as file:

        data = json.load(file)

    zone = np.array(data["zone"], dtype=np.int32)

    return zone


def go_to(mode):

    st.session_state.mode = mode

    st.rerun()


def render_menu():

    st.title("PEOPLE COUNTER")

    if zone_file_is_valid():

        st.success("Zone: Đã có sẵn (zone.json)")

    else:

        st.error("Zone: Chưa có / không hợp lệ -> cần chọn Zone trước khi chạy")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Chọn / Vẽ Zone", use_container_width=True):

            #select_zone_mode mở cửa sổ OpenCV riêng, chạy tới khi nhấn Q
            select_zone_mode(VIDEO_PATH)

            #Rerun để cập nhật lại status Zone sau khi đóng cửa sổ
            st.rerun()

    with col2:

        if st.button("Chạy People Counter", use_container_width=True):

            if not zone_file_is_valid():

                st.warning(
                    "Chưa có zone hoặc zone không hợp lệ (cần ít nhất 3 điểm). "
                    "Vui lòng chọn / vẽ Zone trước."
                )

            else:

                go_to("counter")


def draw_frame(frame, people, zone, person_states):

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

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        #ZONE = cam, OUTSIDE = xanh lá
        if current_state == "ZONE":

            color = (0, 165, 255)

        else:

            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame


def render_counter():

    if st.button("< Quay lại Menu"):

        go_to("menu")

    zone = load_zone()

    col_video, col_info = st.columns([2, 1])

    video_placeholder = col_video.empty()

    summary_placeholder = col_info.empty()
    detail_placeholder = col_info.empty()

    video = cv2.VideoCapture(VIDEO_PATH)

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:

        fps = 30

    set_confirm_time(fps, CONFIRM_SECONDS)

    #Delay giữa các frame để không phát nhanh hơn tốc độ thật của video
    frame_delay = 1.0 / fps

    missing_frames = {}

    try:

        while True:

            ret, frame = video.read()

            if not ret:

                summary_placeholder.info("Video đã kết thúc.")

                break

            people = track_people(frame)

            current_ids = []
            person_states = {}

            for person in people:

                track_id = person[5]

                current_ids.append(track_id)

                #ID xuất hiện ở frame này -> reset missing frames
                missing_frames[track_id] = 0

                state = get_person_state(person, zone)

                person_states[track_id] = state

            #Tăng missing_frames cho ID đã biết mà không xuất hiện ở frame này
            for track_id in list(people_state.keys()):

                if track_id in current_ids:

                    continue

                missing_frames[track_id] = missing_frames.get(track_id, 0) + 1

                if (
                    missing_frames[track_id] >= MAX_MISSING_FRAMES
                    and people_state[track_id]["tracking"] != "LOST"
                ):

                    people_state[track_id]["tracking"] = "LOST"

            people_count = len(current_ids)

            #Entered luôn được tính lại từ people_state, không cộng/trừ tay
            entered_count = sum(
                1 for state in people_state.values() if state["entered"]
            )

            frame = draw_frame(frame, people, zone, person_states)

            #Đổi BGR (OpenCV) -> RGB (Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Placeholder .image() thay thế hoàn toàn ảnh cũ, không bị chồng
            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                width=DISPLAY_WIDTH
            )

            summary_placeholder.markdown(
                f"### People: {people_count}    Entered: {entered_count}"
            )

            #Bảng chi tiết cũng bị thay thế hoàn toàn mỗi lần gọi, không cộng dồn
            if current_ids:

                rows = [
                    {
                        "ID": track_id,
                        "State": person_states[track_id]["current"],
                        "Entered": person_states[track_id]["entered"],
                        "Tracking": person_states[track_id]["tracking"]
                    }
                    for track_id in current_ids
                ]

            else:

                rows = [{"ID": "-", "State": "-", "Entered": "-", "Tracking": "-"}]

            detail_placeholder.dataframe(rows, use_container_width=True, hide_index=True)

            time.sleep(frame_delay)

    finally:

        video.release()


def main():

    st.set_page_config(page_title="People Counter", layout="wide")

    if "mode" not in st.session_state:

        st.session_state.mode = "menu"

    if st.session_state.mode == "menu":

        render_menu()

    elif st.session_state.mode == "counter":

        render_counter()


if __name__ == "__main__":

    main()