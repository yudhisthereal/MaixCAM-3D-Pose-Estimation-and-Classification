# skeleton_saver.py

import csv
import os

class SkeletonSaver:
    def __init__(self):
        self.data_buffer = []

    def add_keypoints(self, frame_id, person_id, keypoints_flat):
        if not keypoints_flat:
            print(f"[DEBUG] No keypoints for frame {frame_id}")
            return

        # Convert flat list to pairs
        pairs = [(keypoints_flat[i], keypoints_flat[i+1]) for i in range(0, len(keypoints_flat), 2)]
        num_points = len(pairs)
        flat_coords = keypoints_flat  # already flattened

        print(f"[DEBUG] Frame {frame_id} – saving {num_points} keypoints")
        self.data_buffer.append([frame_id, person_id] + flat_coords)

    def save_to_csv(self, video_filename):
        """Save buffered keypoints to CSV using video filename as base"""
        base_name = os.path.splitext(video_filename)[0]
        csv_dir = "csv"
        csv_filename = os.path.join(csv_dir, f"{base_name}.csv")

        # Create directory if it doesn't exist
        os.makedirs(csv_dir, exist_ok=True)

        if not self.data_buffer:
            print("[WARNING] No keypoints to save.")
            return

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['frame_id', 'person_id'] + [f'kp{i}_{c}' for i in range(len(self.data_buffer[0]) - 2) for c in ['x', 'y']]
            writer.writerow(header)
            writer.writerows(self.data_buffer)

        print(f"[INFO] Saved keypoints to {csv_filename}")
        self.data_buffer = []  # Clear buffer after save

