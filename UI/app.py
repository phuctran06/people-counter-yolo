import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from detector import track_people, get_person_state, set_confirm_time, people_state
from select_zone import select_zone_mode


#Đường dẫn video và file zone dùng chung cho cả 2 chức năng
VIDEO_PATH = "videos/input.mp4"
ZONE_PATH = "zone.json"

#Số frame cho phép 1 ID biến mất tạm thời trước khi coi là LOST thật
MAX_MISSING_FRAMES = 10

#Thời gian (giây) cần thấy 1 state mới liên tục thì mới tin là transition thật
CONFIRM_SECONDS = 0.5


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


class PeopleCounterApp:

    def __init__(self, root):

        self.root = root
        self.root.title("People Counter")
        self.root.geometry("1100x650")

        self.build_menu()

    def clear_window(self):

        for widget in self.root.winfo_children():

            widget.destroy()

    def build_menu(self):

        self.clear_window()

        title = tk.Label(
            self.root,
            text="PEOPLE COUNTER",
            font=("Arial", 22, "bold")
        )
        title.pack(pady=40)

        select_zone_btn = tk.Button(
            self.root,
            text="Chọn / Vẽ Zone",
            font=("Arial", 14),
            width=28,
            command=self.open_select_zone
        )
        select_zone_btn.pack(pady=10)

        run_btn = tk.Button(
            self.root,
            text="Chạy People Counter",
            font=("Arial", 14),
            width=28,
            command=self.open_counter
        )
        run_btn.pack(pady=10)

        if zone_file_is_valid():

            status_text = "Zone: Đã có sẵn (zone.json)"
            status_color = "green"

        else:

            status_text = "Zone: Chưa có / không hợp lệ -> cần chọn Zone trước"
            status_color = "red"

        status_label = tk.Label(
            self.root,
            text=status_text,
            font=("Arial", 12),
            fg=status_color
        )
        status_label.pack(pady=30)

    def open_select_zone(self):

        #select_zone_mode dùng cửa sổ OpenCV riêng, chạy tới khi người dùng nhấn Q
        select_zone_mode(VIDEO_PATH)

        #Sau khi đóng cửa sổ chọn zone, quay lại menu để cập nhật status
        self.build_menu()

    def open_counter(self):

        if not zone_file_is_valid():

            messagebox.showwarning(
                "Chưa có Zone",
                "Chưa có zone hoặc zone không hợp lệ (cần ít nhất 3 điểm).\n"
                "Vui lòng bấm \"Chọn / Vẽ Zone\" trước khi chạy."
            )

            return

        zone = load_zone()

        self.clear_window()

        CounterView(self.root, self, VIDEO_PATH, zone)


class CounterView:

    def __init__(self, root, app, video_path, zone):

        self.root = root
        self.app = app
        self.zone = zone
        self.running = True

        self.video = cv2.VideoCapture(video_path)

        fps = self.video.get(cv2.CAP_PROP_FPS)

        if fps <= 0:

            fps = 30

        set_confirm_time(fps, CONFIRM_SECONDS)

        self.missing_frames = {}

        self.build_layout()

        self.update_frame()

    def build_layout(self):

        #Thanh trên cùng có nút quay lại menu
        top_bar = tk.Frame(self.root)
        top_bar.pack(side="top", fill="x")

        back_btn = tk.Button(
            top_bar,
            text="< Quay lại Menu",
            command=self.stop_and_back
        )
        back_btn.pack(side="left", padx=10, pady=10)

        main_frame = tk.Frame(self.root)
        main_frame.pack(side="top", fill="both", expand=True)

        #Bên trái: video
        self.video_label = tk.Label(main_frame)
        self.video_label.pack(side="left", padx=10, pady=10)

        #Bên phải: info panel
        info_frame = tk.Frame(main_frame, width=320)
        info_frame.pack(side="right", fill="both", padx=10, pady=10)

        self.summary_label = tk.Label(
            info_frame,
            text="People: 0    Entered: 0",
            font=("Arial", 14, "bold"),
            justify="left"
        )
        self.summary_label.pack(anchor="w", pady=(0, 10))

        detail_title = tk.Label(
            info_frame,
            text="Chi tiết từng người trong frame:",
            font=("Arial", 12, "bold")
        )
        detail_title.pack(anchor="w")

        self.detail_text = tk.Text(
            info_frame,
            width=38,
            height=28,
            font=("Consolas", 10)
        )
        self.detail_text.pack(anchor="w", fill="both", expand=True)

    def update_frame(self):

        if not self.running:

            return

        ret, frame = self.video.read()

        if not ret:

            messagebox.showinfo("Kết thúc", "Video đã kết thúc.")

            self.stop_and_back()

            return

        people = track_people(frame)

        current_ids = []
        person_states = {}

        for person in people:

            track_id = person[5]

            current_ids.append(track_id)

            #ID xuất hiện ở frame này -> reset missing frames
            self.missing_frames[track_id] = 0

            state = get_person_state(person, self.zone)

            person_states[track_id] = state

        #Tăng missing_frames cho ID đã biết mà không xuất hiện ở frame này
        for track_id in list(people_state.keys()):

            if track_id in current_ids:

                continue

            self.missing_frames[track_id] = self.missing_frames.get(track_id, 0) + 1

            if (
                self.missing_frames[track_id] >= MAX_MISSING_FRAMES
                and people_state[track_id]["tracking"] != "LOST"
            ):

                people_state[track_id]["tracking"] = "LOST"

        people_count = len(current_ids)

        #Entered luôn được tính lại từ people_state, không cộng/trừ tay
        entered_count = sum(
            1 for state in people_state.values() if state["entered"]
        )

        frame = self.draw_frame(frame, people, person_states)

        self.render_video(frame)

        self.render_info(people_count, entered_count, current_ids, person_states)

        #~30fps cho UI, không cần khớp chính xác FPS gốc của video
        self.root.after(15, self.update_frame)

    def draw_frame(self, frame, people, person_states):

        #Vẽ zone
        cv2.polylines(
            frame,
            [self.zone],
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

    def render_video(self, frame):

        #Đổi màu BGR (OpenCV) sang RGB (Tkinter/PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame_rgb)

        img = img.resize((720, 480))

        imgtk = ImageTk.PhotoImage(image=img)

        #Giữ reference để tránh bị garbage collect mất ảnh
        self.video_label.imgtk = imgtk

        self.video_label.configure(image=imgtk)

    def render_info(self, people_count, entered_count, current_ids, person_states):

        #Label chỉ cần .config(text=...) là tự thay nội dung cũ, không bị chồng chữ
        self.summary_label.config(
            text=f"People: {people_count}    Entered: {entered_count}"
        )

        #Text widget: xóa toàn bộ nội dung cũ rồi ghi lại danh sách mới
        self.detail_text.delete("1.0", "end")

        if not current_ids:

            self.detail_text.insert("end", "(Không có ai trong frame)")

            return

        header = f"{'ID':<5}{'State':<9}{'Entered':<9}{'Tracking'}\n"

        self.detail_text.insert("end", header)
        self.detail_text.insert("end", "-" * 32 + "\n")

        for track_id in current_ids:

            state = person_states[track_id]

            line = (
                f"{track_id:<5}"
                f"{state['current']:<9}"
                f"{str(state['entered']):<9}"
                f"{state['tracking']}\n"
            )

            self.detail_text.insert("end", line)

    def stop_and_back(self):

        self.running = False

        self.video.release()

        self.app.build_menu()


def main():

    root = tk.Tk()

    app = PeopleCounterApp(root)

    root.mainloop()


if __name__ == "__main__":

    main()