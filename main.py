from maix import camera, display, app, nn, image, time
from pose.pose_estimation import PoseEstimation
from tools.wifi_connect import connect_wifi
from tools.video_record import VideoRecorder
from tools.time_utils import get_timestamp_str

import numpy as np
import os

SSID = "MaixCAM-Wifi"
PASSWORD = "maixcamwifi"
connect_wifi(SSID, PASSWORD)

detector = nn.YOLO11(model="/root/models/yolo11n_pose.mud", dual_buff=False)
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(), fps=60)
disp = display.Display()

pose_estimator = PoseEstimation()

image.load_font("sourcehansans", "/maixapp/share/font/SourceHanSansCN-Regular.otf", size=32)
image.set_default_font("sourcehansans")

# === Recorder setup ===
record_period = 10000 # 10 seconds
video_dir = "/root/recordings"
os.makedirs(video_dir, exist_ok=True)

recorder = VideoRecorder()

def start_new_recording():
    timestamp = get_timestamp_str()
    video_path = os.path.join(video_dir, f"{timestamp}.mp4")
    recorder.start(video_path, detector.input_width(), detector.input_height())
    return time.ticks_ms()

# Start first recording
last_rotate_ms = start_new_recording()

def to_keypoints_np(obj_points):
    keypoints = np.array(obj_points)
    return keypoints.reshape((-1, 2))

# === Main loop ===
while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45, keypoint_th=0.5)
    for obj in objs:
        msg = f'[{obj.score:.2f}], {pose_estimator.evaluate_pose(to_keypoints_np(obj.points))}'
        img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED, scale=0.5)
        detector.draw_pose(img, obj.points, 8 if detector.input_width() > 480 else 4, image.COLOR_RED)

    recorder.add_frame(img)
    disp.show(img)

    # 🔁 Rotate recording every 10 seconds
    now = time.ticks_ms()
    if now - last_rotate_ms >= record_period:
        recorder.end()
        last_rotate_ms = start_new_recording()

# Final cleanup
recorder.end()
