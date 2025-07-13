from maix import camera, display, app, nn, image
from pose.pose_estimation import PoseEstimation
from tools.wifi_connect import connect_wifi
from tools.video_record import VideoRecorder

import numpy as np

SSID="MaixCAM-Wifi"
PASSWORD="maixcamwifi"
connect_wifi(SSID, PASSWORD)

detector = nn.YOLO11(model="/root/models/yolo11n_pose.mud", dual_buff = False)
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(), fps=60)
disp = display.Display()

def to_keypoints_np(obj_points):
    keypoints = np.array(obj_points)
    keypoints = keypoints.reshape((-1, 2))
    # print("kps: ", keypoints)
    return keypoints

pose_estimator = PoseEstimation()

image.load_font("sourcehansans", "/maixapp/share/font/SourceHanSansCN-Regular.otf", size = 32)
# print("fonts:", image.fonts())
image.set_default_font("sourcehansans")
while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45, keypoint_th = 0.5)
    for obj in objs:
        # img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'[{obj.score:.2f}], {pose_estimator.evaluate_pose(to_keypoints_np(obj.points))}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED, scale=0.5)
        detector.draw_pose(img, obj.points, 8 if detector.input_width() > 480 else 4, image.COLOR_RED)
    disp.show(img)
