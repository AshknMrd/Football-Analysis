from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import argparse
import tomllib


PATH_KEYS = {
    "input_video",
    "output_video",
    "model",
    "track_stub",
    "camera_movement_stub",
    "images_dir",
    "models_dir",
    "input_videos_dir",
    "output_videos_dir",
    "stubs_dir",
    "notebooks_dir",
    "dataset_yaml",
}


class Config(SimpleNamespace):
    """Attribute-style access for values loaded from config.toml."""


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return {key: _to_namespace(item) for key, item in value.items()}
        return Config(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(config_path: str | Path) -> Config:
    config_file = Path(config_path).resolve()
    with config_file.open("rb") as file:
        config = tomllib.load(file)

    base_dir = config_file.parent
    config["config_path"] = config_file

    for key in PATH_KEYS:
        if key in config["paths"]:
            config["paths"][key] = _resolve_path(config["paths"][key], base_dir)

    project = config.get("yolo_inference", {}).get("project")
    if project:
        config["yolo_inference"]["project"] = _resolve_path(project, base_dir)

    overrides = config.get("team_assignment", {}).get("goalkeeper_team_overrides", {})
    config["team_assignment"]["goalkeeper_team_overrides"] = {
        int(player_id): int(team_id)
        for player_id, team_id in overrides.items()
    }

    return _to_namespace(config)


def run_pipeline(config) -> None:
    import numpy as np

    from .camera_movement_estimator import CameraMovementEstimator
    from .player_ball_assigner import PlayerBallAssigner
    from .speed_and_distance_estimator import SpeedAndDistanceEstimator
    from .team_assigner import TeamAssigner
    from .trackers import Tracker
    from .utils import read_video, save_video
    from .view_transformer import ViewTransformer

    video_frames = read_video(config.paths.input_video)
    if not video_frames:
        raise ValueError(f"No frames could be read from input video: {config.paths.input_video}")

    tracker = Tracker(
        config.paths.model,
        batch_size=config.tracking.batch_size,
        confidence=config.tracking.confidence,
        convert_goalkeepers_to_players=config.tracking.convert_goalkeepers_to_players,
    )

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=config.tracking.read_from_stub,
        stub_path=config.paths.track_stub,
        write_to_stub=config.tracking.write_to_stub,
    )
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(
        video_frames[0],
        minimum_distance=config.camera_movement.minimum_distance,
        feature_mask_columns=config.camera_movement.feature_mask_columns,
        lk_win_size=config.camera_movement.lk_win_size,
        lk_max_level=config.camera_movement.lk_max_level,
        lk_criteria_count=config.camera_movement.lk_criteria_count,
        lk_criteria_epsilon=config.camera_movement.lk_criteria_epsilon,
        feature_max_corners=config.camera_movement.feature_max_corners,
        feature_quality_level=config.camera_movement.feature_quality_level,
        feature_min_distance=config.camera_movement.feature_min_distance,
        feature_block_size=config.camera_movement.feature_block_size,
    )
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=config.camera_movement.read_from_stub,
        stub_path=config.paths.camera_movement_stub,
        write_to_stub=config.camera_movement.write_to_stub,
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer(
        court_width=config.view_transform.court_width,
        court_length=config.view_transform.court_length,
        pixel_vertices=config.view_transform.pixel_vertices,
    )
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistanceEstimator(
        frame_window=config.speed_distance.frame_window,
        frame_rate=config.speed_distance.frame_rate,
    )
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner(
        crop_kmeans_n_init=config.team_assignment.crop_kmeans_n_init,
        team_kmeans_n_init=config.team_assignment.team_kmeans_n_init,
        goalkeeper_team_overrides=config.team_assignment.goalkeeper_team_overrides,
    )
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner(
        max_player_ball_distance=config.player_ball.max_player_ball_distance,
    )
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame,
    )
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(
        output_video_frames,
        config.paths.output_video,
        fps=config.video.output_fps,
        codec=config.video.output_codec,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the football video analysis pipeline.")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to the TOML configuration file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_pipeline(config)
