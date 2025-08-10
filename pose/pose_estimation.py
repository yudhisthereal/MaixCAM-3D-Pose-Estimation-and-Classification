import numpy as np
from collections import deque

class PoseEstimation:
    """Very small pose classifier: returns one of 'standing', 'sitting', 'bending_down', 'lying_down', or None

    Changes:
    - Reject frames with incomplete keypoints by checking a missing-value sentinel (default -1).
    - Append to the smoothing deque only when the current frame is complete.
    - Return None (and set self.status = []) when keypoints are incomplete or angles cannot be computed.
    """
    def __init__(self, keypoints_window_size=5, missing_value=-1):
        self.keypoints_map_deque = deque(maxlen=keypoints_window_size)
        self.status = []
        self.missing_value = missing_value

    def feed_keypoints_17(self, keypoints_17):
        # keypoints_17: flattened list/array [x0,y0, x1,y1, ..., x16,y16]
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

        return self.feed_keypoints_map(kp_map)

    def _is_frame_complete(self, keypoints_map):
        """Return True if none of the required keypoints contain the missing_value sentinel."""
        for k, v in keypoints_map.items():
            # v is an array-like [x, y]
            if v is None:
                return False
            # check both coordinates for sentinel
            if v[0] == self.missing_value or v[1] == self.missing_value:
                return False
        return True

    def feed_keypoints_map(self, keypoints_map):
        # If current frame is incomplete, clear status and do not append — return None
        if not self._is_frame_complete(keypoints_map):
            self.status = []
            return None

        # append the verified-complete frame for temporal smoothing
        self.keypoints_map_deque.append(keypoints_map)

        # compute averaged keypoints over the deque
        km = {
            key: sum(d[key] for d in self.keypoints_map_deque) / len(self.keypoints_map_deque)
            for key in self.keypoints_map_deque[0].keys()
        }

        # compute centers
        shoulder_center = (km['Left Shoulder'] + km['Right Shoulder']) / 2.0
        hip_center = (km['Left Hip'] + km['Right Hip']) / 2.0
        knee_center = (km['Left Knee'] + km['Right Knee']) / 2.0

        torso_vec = shoulder_center - hip_center
        thigh_vec = knee_center - hip_center

        up_vector = np.array([0.0, -1.0])

        # safe angle computation: if vector norm is zero, return None
        torso_norm = np.linalg.norm(torso_vec)
        thigh_norm = np.linalg.norm(thigh_vec)
        if torso_norm == 0 or thigh_norm == 0:
            self.status = []
            return None

        torso_angle = np.degrees(np.arccos(np.clip(
            np.dot(torso_vec, up_vector) / (torso_norm * np.linalg.norm(up_vector)), -1.0, 1.0)))

        thigh_angle = np.degrees(np.arccos(np.clip(
            np.dot(thigh_vec, up_vector) / (thigh_norm * np.linalg.norm(up_vector)), -1.0, 1.0)))

        # convert thigh angle to "uprightness" where smaller is more upright
        thigh_uprightness = abs(thigh_angle - 180.0)

        # simple thresholds (tweakable)
        if torso_angle < 30 and thigh_uprightness < 40:
            label = "standing"
        elif torso_angle < 30 and thigh_uprightness >= 40:
            label = "sitting"
        elif 30 <= torso_angle < 80 and thigh_uprightness < 60:
            label = "bending_down"
        else:
            label = "lying_down"

        self.status = [label]
        return label

    def evaluate_pose(self, keypoints):
        # returns label string or None
        res = self.feed_keypoints_17(keypoints)
        if res is None:
            return None
        return self.status[0] if self.status else None
