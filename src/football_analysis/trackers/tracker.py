from pathlib import Path
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from ..utils import get_bbox_width, get_center_of_bbox, get_foot_position


class Tracker:
    def __init__(
        self,
        model_path,
        batch_size=20,
        confidence=0.1,
        convert_goalkeepers_to_players=True,
    ):
        self.model = YOLO(str(model_path))
        self.tracker = sv.ByteTrack()
        self.batch_size = batch_size
        self.confidence = confidence
        self.convert_goalkeepers_to_players = convert_goalkeepers_to_players

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object_name == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        return [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(
                frames[i : i + self.batch_size],
                conf=self.confidence,
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, write_to_stub=True):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as file:
                tracks = pickle.load(file)
            if len(tracks.get("players", [])) == len(frames):
                return tracks
            print(
                f"Ignoring tracking stub with {len(tracks.get('players', []))} frames "
                f"for video with {len(frames)} frames: {stub_path}"
            )

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            if self.convert_goalkeepers_to_players:
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if write_to_stub and stub_path is not None:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as file:
                pickle.dump(tracks, file)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()

        x_max, y_max = frame.shape[1], frame.shape[0]
        print_rectangle = (0.775 * x_max, 0.088 * y_max)
        cv2.rectangle(
            overlay,
            (int(print_rectangle[0]), int(print_rectangle[1])),
            (x_max - 10, int(print_rectangle[1]) + 70),
            (0, 0, 0),
            -1,
        )

        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f"Ball Team1: {team_1 * 100:.1f}%",
            (int(print_rectangle[0]) + 20, int(print_rectangle[1]) + 20),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            f"Ball Team2: {team_2 * 100:.1f}%",
            (int(print_rectangle[0]) + 20, int(print_rectangle[1]) + 50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 255, 255),
            3,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
