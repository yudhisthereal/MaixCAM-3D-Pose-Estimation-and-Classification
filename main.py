from maix import camera, display, app, nn, image, time, tracker
from pose.pose_estimation import PoseEstimation
from tools.wifi_connect import connect_wifi
from tools.video_record import VideoRecorder
from tools.time_utils import get_timestamp_str
from tools.skeleton_saver import SkeletonSaver2D
from pose.judge_fall import get_fall_info
from tools import web_server
import queue
import numpy as np
import os

# Image Paths
BACKGROUND_PATH = "/root/static/background.jpg"

# Wi-Fi Setup
# SSID = "MaixCAM-Wifi"
# PASSWORD = "maixcamwifi"
SSID = "ROBOTIIK"
PASSWORD = "81895656"
server_ip = connect_wifi(SSID, PASSWORD)

# Web Server Setup
web_server.start_servers()
print("\nServer started. Connect to MaixCAM in your browser:")
print(f"   → http://{server_ip}:{web_server.HTTP_PORT}/")

# Ensure static dir exist
os.makedirs("/root/static", exist_ok=True)

detector = nn.YOLO11(model="/root/models/yolo11n_pose.mud", dual_buff=False)
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(), fps=60)
disp = display.Display()

pose_estimator = PoseEstimation()

image.load_font("sourcehansans", "/maixapp/share/font/SourceHanSansCN-Regular.otf", size=32)
image.set_default_font("sourcehansans")

# Skeleton Saver and Recorder setup
record_period = 10000 # 10 seconds
video_dir = "/root/recordings"
os.makedirs(video_dir, exist_ok=True)

recorder = VideoRecorder()
skeleton_saver_2d = SkeletonSaver2D()
frame_id = 0 # the frame ID of the current recording (resets every time a new recording is started)

def start_new_recording():
    global frame_id
    
    timestamp = get_timestamp_str()
    video_path = os.path.join(video_dir, f"{timestamp}.mp4")
    recorder.start(video_path, detector.input_width(), detector.input_height())
    skeleton_saver_2d.start_new_log(timestamp)
    frame_id = 0
    
    return time.ticks_ms()

# Start first recording
last_rotate_ms = start_new_recording()

def to_keypoints_np(obj_points):
    keypoints = np.array(obj_points)
    return keypoints.reshape(-1, 2)

# Tracker and fall detection setup
fallParam = {
    "v_bbox_y": 0.43,
    "angle": 70
}
fps = cam.fps()
queue_size = 5

online_targets = {
    "id": [],
    "bbox": [],
    "points": []
}

fall_down = False
fall_ids = set()

# Tracking params
max_lost_buff_time = 30
track_thresh = 0.4
high_thresh = 0.6
match_thresh = 0.8
max_history_num = 5
valid_class_id = [0]
tracker0 = tracker.ByteTracker(max_lost_buff_time, track_thresh, high_thresh, match_thresh, max_history_num)

def yolo_objs_to_tracker_objs(objs, valid_class_id=[0]):
    out = []
    for obj in objs:
        if obj.class_id in valid_class_id:
            out.append(tracker.Object(obj.x, obj.y, obj.w, obj.h, obj.class_id, obj.score))
    return out

# === Main loop ===
while not app.need_exit():
    raw_img = cam.read()
    flags = web_server.get_control_flags()

    if flags["show_raw"]:
        img = raw_img.copy()
    else:
        if os.path.exists(BACKGROUND_PATH):
            img = image.load(BACKGROUND_PATH, format=image.Format.FMT_RGB888)
        else:
            img = raw_img.copy()  # fallback

    objs = detector.detect(raw_img, conf_th=0.5, iou_th=0.45, keypoint_th=0.5)
    out_bbox = yolo_objs_to_tracker_objs(objs)
    tracks = tracker0.update(out_bbox)

    frame_id += 1
    
    for track in tracks:
        if track.lost:
            continue
        for tracker_obj in track.history[-1:]:
            for obj in objs:
                if abs(obj.x - tracker_obj.x) < 10 and abs(obj.y - tracker_obj.y) < 10:
                    keypoints_np = to_keypoints_np(obj.points)

                    # Assign ID
                    if track.id not in online_targets["id"]:
                        online_targets["id"].append(track.id)
                        online_targets["bbox"].append(queue.Queue(maxsize=queue_size))
                        online_targets["points"].append(queue.Queue(maxsize=queue_size))

                    idx = online_targets["id"].index(track.id)

                    # Add bbox and points to queue
                    if online_targets["bbox"][idx].qsize() >= queue_size:
                        online_targets["bbox"][idx].get()
                        online_targets["points"][idx].get()
                    online_targets["bbox"][idx].put([tracker_obj.x, tracker_obj.y, tracker_obj.w, tracker_obj.h])
                    online_targets["points"][idx].put(obj.points)

                    # Call fall detection when queue is full
                    if online_targets["bbox"][idx].qsize() == queue_size:
                        if get_fall_info(tracker_obj, online_targets, idx, fallParam, queue_size, fps):
                            fall_ids.add(track.id)
                        elif track.id in fall_ids:
                            fall_ids.remove(track.id)

                    # Draw
                    status_str = pose_estimator.evaluate_pose(keypoints_np)
                    if track.id in fall_ids:
                        msg = f"[{track.id}] FALL"
                        color = image.COLOR_RED
                    else:
                        msg = f"[{track.id}] {status_str}"
                        color = image.COLOR_GREEN

                    img.draw_string(int(tracker_obj.x), int(tracker_obj.y), msg, color=color, scale=0.5)
                    detector.draw_pose(img, obj.points, 8 if detector.input_width() > 480 else 4, color=color)
                    
                    skeleton_saver_2d.add_keypoints(frame_id, track.id, obj.points, 1 if track.id in fall_ids else 0)
                    
                    break  # no need to check other objs


    if recorder.is_active:
        recorder.add_frame(img)
    disp.show(img)
    
    web_server.send_frame(img)

    flags = web_server.get_control_flags()
    if flags["record"] and not recorder.is_active:
        last_rotate_ms = start_new_recording()
    elif not flags["record"] and recorder.is_active:
        recorder.end()

    if flags["set_background"]:
        web_server.confirm_background(BACKGROUND_PATH)
        web_server.reset_set_background_flag()


    # Rotate recording every 10 seconds
    now = time.ticks_ms()
    if now - last_rotate_ms >= record_period:
        # Save vid recording and skeleton csv
        recorder.end()
        skeleton_saver_2d.save_to_csv()
        
        # Start new rec
        last_rotate_ms = start_new_recording()


# Final cleanup
recorder.end()
