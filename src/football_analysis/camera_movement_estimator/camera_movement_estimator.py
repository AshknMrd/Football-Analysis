from pathlib import Path
import os
import pickle

import cv2
import numpy as np

from ..utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(
        self,
        frame,
        minimum_distance=5,
        feature_mask_columns=((0, 20), (900, 1050)),
        lk_win_size=(15, 15),
        lk_max_level=2,
        lk_criteria_count=10,
        lk_criteria_epsilon=0.03,
        feature_max_corners=100,
        feature_quality_level=0.3,
        feature_min_distance=3,
        feature_block_size=7,
    ):
        self.minimum_distance = minimum_distance
        self.lk_params = dict(
            winSize=tuple(lk_win_size),
            maxLevel=lk_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                lk_criteria_count,
                lk_criteria_epsilon,
            ),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        for start, end in feature_mask_columns:
            mask_features[:, int(start) : int(end)] = 1

        self.features = dict(
            maxCorners=feature_max_corners,
            qualityLevel=feature_quality_level,
            minDistance=feature_min_distance,
            blockSize=feature_block_size,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object_name][frame_num][track_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None, write_to_stub=True):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as file:
                camera_movement = pickle.load(file)
            if len(camera_movement) == len(frames):
                return camera_movement
            print(
                f"Ignoring camera movement stub with {len(camera_movement)} frames "
                f"for video with {len(frames)} frames: {stub_path}"
            )

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                old_features,
                None,
                **self.lk_params,
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for new, old in zip(new_features, old_features):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point,
                        new_features_point,
                    )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if write_to_stub and stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as file:
                pickle.dump(camera_movement, file)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()

            x_max, y_max = frame.shape[1], frame.shape[0]
            print_rectangle = (0.633 * x_max, 0.088 * y_max)
            cv2.rectangle(
                overlay,
                (int(print_rectangle[0]), int(print_rectangle[1])),
                (int(print_rectangle[0]) + 200, int(print_rectangle[1]) + 70),
                (0, 0, 0),
                -1,
            )

            alpha = 0.8
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera-X: {x_movement:.1f}",
                (int(print_rectangle[0]) + 10, int(print_rectangle[1]) + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                3,
            )
            frame = cv2.putText(
                frame,
                f"Camera-Y: {y_movement:.1f}",
                (int(print_rectangle[0]) + 10, int(print_rectangle[1]) + 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                3,
            )

            output_frames.append(frame)

        return output_frames
