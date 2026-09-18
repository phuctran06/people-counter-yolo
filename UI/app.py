import os
import sys
import json
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from detector import track_people, get_person_state, set_confirm_time, people_state

VIDEO_PATH = os.path.join(PROJECT_ROOT, "videos", "input.mp4")
ZONE_PATH = os.path.join(PROJECT_ROOT, "zone.json")

MAX_MISSING_FRAMES = 10
CONFIRM_SECONDS = 0.5
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
    return len(zone_points) >= 3


def load_zone():
    with open(ZONE_PATH, "r") as file:
        data = json.load(file)
    zone = np.array(data["zone"], dtype=np.int32)
    return zone


def save_zone_points(points):
    data = {"zone": points}
    with open(ZONE_PATH, "w") as file:
        json.dump(data, file, indent=4)


@st.cache_data
def get_first_frame(video_path):
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    video.release()
    if not ret:
        return None
    return frame


def draw_zone_preview(frame_rgb, points):
    preview = frame_rgb.copy()
    for point in points:
        cv2.circle(preview, point, 6, (255, 0, 0), -1)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(preview, points[i], points[i + 1], (255, 0, 0), 2)
    if len(points) >= 3:
        cv2.line(preview, points[-1], points[0], (255, 0, 0), 2)
    return preview


def render_select_zone():
    st.header("Chọn / Vẽ Zone")
    st.write(
        "Click vào ảnh để thêm điểm cho zone (cần ít nhất 3 điểm). "
        "Điểm sẽ được nối theo thứ tự bạn click."
    )

    if "zone_points" not in st.session_state:
        if zone_file_is_valid():
            with open(ZONE_PATH, "r") as file:
                data = json.load(file)
            st.session_state.zone_points = [tuple(p) for p in data["zone"]]
        else:
            st.session_state.zone_points = []

    if "last_click_value" not in st.session_state:
        st.session_state.last_click_value = None

    frame = get_first_frame(VIDEO_PATH)
    if frame is None:
        st.error("Không đọc được video để lấy frame đầu tiên. Kiểm tra lại VIDEO_PATH.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preview = draw_zone_preview(frame_rgb, st.session_state.zone_points)
    pil_image = Image.fromarray(preview)

    value = streamlit_image_coordinates(
        pil_image,
        key="zone_click",
        width=DISPLAY_WIDTH
    )

    if value is not None and value != st.session_state.last_click_value:
        st.session_state.last_click_value = value
        point = (int(value["x"]), int(value["y"]))
        st.session_state.zone_points.append(point)
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Xóa điểm cuối", use_container_width=True):
            if st.session_state.zone_points:
                st.session_state.zone_points.pop()
            st.rerun()
    with col2:
        if st.button("Xóa hết", use_container_width=True):
            st.session_state.zone_points = []
            st.rerun()
    with col3:
        can_save = len(st.session_state.zone_points) >= 3
        if st.button("Lưu Zone", type="primary", use_container_width=True, disabled=not can_save):
            save_zone_points(st.session_state.zone_points)
            st.success("Đã lưu zone.json!")

    st.caption(f"Số điểm hiện tại: {len(st.session_state.zone_points)}")


def draw_frame(frame, people, zone, person_states):
    cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)

    for person in people:
        x1, y1, x2, y2, confidence, track_id = person
        state = person_states[track_id]
        current_state = state["current"]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        color = (0, 165, 255) if current_state == "ZONE" else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame


def render_counter():
    st.header("Chạy People Counter")

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
                missing_frames[track_id] = 0
                state = get_person_state(person, zone)
                person_states[track_id] = state

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
            entered_count = sum(
                1 for state in people_state.values() if state["entered"]
            )

            frame = draw_frame(frame, people, zone, person_states)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(frame_rgb, channels="RGB", width=DISPLAY_WIDTH)
            summary_placeholder.markdown(
                f"### People: {people_count}    Entered: {entered_count}"
            )

            if current_ids:
                rows = [
                    {
                        "ID": track_id,
                        "State": person_states[track_id]["current"],
                        "Entered": person_states[track_id]["entered"],
                        "Tracking": person_states[track_id]["tracking"],
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

    st.sidebar.title("People Counter")
    mode = st.sidebar.radio("Chức năng", ["Chạy People Counter", "Chọn / Vẽ Zone"])

    if zone_file_is_valid():
        st.sidebar.success("Zone: Đã có sẵn")
    else:
        st.sidebar.error("Zone: Chưa có / không hợp lệ")

    if mode == "Chọn / Vẽ Zone":
        render_select_zone()
    else:
        if not zone_file_is_valid():
            st.warning(
                "Chưa có zo ne hoặc zone không hợp lệ (cần ít nhất 3 điểm). "
                "Vui lòng chọn \"Chọn / Vẽ Zone\" ở sidebar trước."
                
            )
        else:
            render_counter()


if __name__ == "__main__":

    main()
