import numpy as np
from collections import deque

class PoseEstimation:
    def __init__(self, keypoints_window_size=5):
        self.keypoints_map_deque = deque(maxlen=keypoints_window_size)
        self.status = []

    def feed_keypoints_17(self, keypoints_17):
        keypoints = np.array(keypoints_17).reshape((-1, 2))
        assert keypoints.shape == (17, 2)

        kp_map = {
            'Left Shoulder': keypoints[5],
            'Right Shoulder': keypoints[6],
            'Left Hip': keypoints[11],
            'Right Hip': keypoints[12],
            'Left Knee': keypoints[13],
            'Right Knee': keypoints[14]
        }

        self.feed_keypoints_map(kp_map)

    def feed_keypoints_map(self, keypoints_map):
        self.keypoints_map_deque.append(keypoints_map)

        km = {
            key: sum(d[key] for d in self.keypoints_map_deque) / len(self.keypoints_map_deque)
            for key in self.keypoints_map_deque[0].keys()
        }

        status = []

        shoulder_center = (km['Left Shoulder'] + km['Right Shoulder']) / 2
        hip_center = (km['Left Hip'] + km['Right Hip']) / 2
        knee_center = (km['Left Knee'] + km['Right Knee']) / 2

        torso_vec = shoulder_center - hip_center
        thigh_vec = knee_center - hip_center

        up_vector = np.array([0, -1])

        torso_norm = np.linalg.norm(torso_vec)
        thigh_norm = np.linalg.norm(thigh_vec)
        if torso_norm == 0 or thigh_norm == 0:
            return "None" # Incomplete keypoints
        else:
            torso_angle = np.degrees(np.arccos(np.clip(
                np.dot(torso_vec, up_vector) / (np.linalg.norm(torso_vec) * np.linalg.norm(up_vector)), -1.0, 1.0)))
            thigh_angle = np.degrees(np.arccos(np.clip(
                np.dot(thigh_vec, up_vector) / (np.linalg.norm(thigh_vec) * np.linalg.norm(up_vector)), -1.0, 1.0)))

        thigh_angle = abs (thigh_angle - 180)

        if torso_angle < 30 and thigh_angle < 40:
            status.append("Standing")
        elif torso_angle < 30 and thigh_angle >= 40:
            status.append(f"Sitting")
        elif 30 <= torso_angle < 80 and thigh_angle >= 70:
            status.append("Bending Down")
        else:
            status.append("Lying Down")

        self.status = status

    def evaluate_pose(self, keypoints):
        self.feed_keypoints_17(keypoints)
        return self.status[0] if self.status else "unknown"
