# Football Video Analysis with YOLO and OpenCV

This project analyzes a football match video and produces an annotated output video with detected players, referees, the ball, team assignments, ball possession, camera movement, and player speed/distance estimates.

The main pipeline is implemented in `main.py`. It uses a trained YOLO model for object detection, ByteTrack for object tracking, OpenCV for video processing and camera-motion estimation, and K-Means clustering for team-color assignment.

## Overview

Given an input video, the project:

- Reads frames from `input_videos/08fd33_4.mp4`.
- Detects football objects using the YOLO weights at `models/yolo5_trained/best.pt`.
- Tracks players and referees across frames with ByteTrack.
- Treats goalkeeper detections as players during tracking, as implemented in `trackers/tracker.py`.
- Interpolates missing ball detections with pandas.
- Assigns players to two teams using shirt-color clustering.
- Assigns ball possession to the nearest player when the ball is within a 70-pixel threshold.
- Estimates camera movement using Lucas-Kanade optical flow on selected fixed regions of the frame.
- Applies a perspective transformation to map selected pitch coordinates into a real-world coordinate space.
- Estimates and draws player speed and cumulative distance.
- Saves the annotated output to `output_videos/output_video.mp4`.

## Dataset

The training notebook uses the Roboflow football-player detection dataset, `football-players-detection-1`, to fine-tune the YOLOv5 model. It includes 663 images across four classes: `ball`, `goalkeeper`, `player`, and `referee`. The local dataset split contains 612 training images, 38 validation images, and 13 test images.

## Model and Methods Deatail:

Since the initial YOLO inference was not accurate enough for detecting players and the ball on the pitch, so YOLOv5 was fine-tuned on a football dataset from Roboflow to detect players, referees, goalkeepers, and the ball. The trained model improves ball detection and helps exclude people outside the pitch.

The next step is to track detected objects consistently across the full video. In several frames, the goalkeeper is detected as a player or switches between the goalkeeper and player classes. Because goalkeeper-specific statistics are not part of this analysis, goalkeeper detections are treated as players.

#### `trackers`

The tracker handles object tracking and makes the output annotations easier to follow. In `tracker.py`, it:

- Tracks detected objects across video frames.
- Draws a red ellipse under each player, scaled to the width of the player's bounding box.
- Draws a yellow ellipse under each referee.
- Displays each player's tracking ID below the player.
- Draws a pointer on the ball.

#### `team_assigner`

The team assigner separates players into two teams based on shirt color. The process is:

- Crop each player using the detected bounding box.
- Use the top half of the crop, where the shirt is usually most visible.
- Cluster the top-half image into background and shirt-color regions using K-Means.
- Extract the player's shirt color from the segmented crop.
- Cluster all player shirt colors into two teams using K-Means.
- Assign each player a team ID and reuse that assignment for the same player ID in later frames.

<table>
  <tr>
    <td align="center">
      <img src="images/player_cropped.png" width="150"><br>
      <em style="font-size:12px;">Cropped player image.</em>
    </td>
    <td align="center">
      <img src="images/player_half_top_cropped.png" width="150"><br>
      <em style="font-size:12px;">Top-half crop used to focus on the shirt color.</em>
    </td>
    <td align="center">
      <img src="images/player_background_segmented.png" width="150"><br>
      <em style="font-size:12px;">Separate shirt-color from the background (K-Means segmentation)</em>
    </td>
  </tr>
</table>

Because the ball is not detected in every frame, missing ball positions are interpolated between known detections.

#### `player_ball_assigner`

After interpolating missing ball detections, the player-ball assigner identifies which player is carrying the ball in each frame and marks that player with a red triangle.

- If the ball is more than 70 pixels away from the closest player, no player is assigned to the ball.
- A ball-control percentage box is added to the top-right of the frame.

For the current analysis, one goalkeeper track is manually assigned to team 1:

```Python
if player_id == 91:  # if goalkeeper assign to team 1
    team_id = 1
```

#### `camera_movement_estimator`

Because the camera moves during the match, player bounding boxes can shift even when the players themselves are not moving. The camera movement estimator compensates for this movement before calculating player motion.

The estimator:

- Selects fixed regions of the frame, such as stable pitch or stadium features.
- Tracks those fixed features with optical flow to estimate camera movement.
- Saves the computed movement data as a pickle file in `stubs` so it can be reused.
- Displays live camera movement values in the top-right of the output frame.
- Adjusts player positions to make movement estimates more robust to camera motion.

<p align="center">
  <tr>
    <td align="center">
      <img src="images/sample_annotations.png" width="500"><br>
      <em style="font-size:12px;">Sample output annotations showing players, referees, the ball, ball possession, speed, and distance.</em>
    </td>
  </tr>
</p>

#### Perspective Transformation: `view_transformer`

After compensating for camera movement, the view transformer estimates a real-world position for tracked players. It uses the pitch dimensions and a selected area of the pitch to compute a perspective transformation.

- The pitch size is treated as 105 x 68 meters.
- A visible region of the pitch is mapped to real-world coordinates.
- The transformed positions are then used to calculate player speed and distance in meters.

<p align="center">
  <tr>
    <td align="center">
      <img src="images/perspective_transformation.png" width="600"><br>
    </td>
  </tr>
</p>

#### `speed_and_distance_estimator`

The speed and distance estimator calculates each player's speed and cumulative running distance, then displays those values below the player's position in the output video.


#### Summary:
The runtime pipeline in `main.py` uses:
- `ultralytics.YOLO` with `models/yolo5_trained/best.pt`
- `supervision.ByteTrack` for tracking detected players and referees
- OpenCV drawing utilities for frame annotations
- OpenCV Lucas-Kanade optical flow for camera movement estimation
- OpenCV perspective transformation for projected position estimates
- scikit-learn `KMeans` for team-color clustering
- pandas interpolation for missing ball bounding boxes
The models and the app can be easily adapted to newer versions of YOLO, such as YOLO26.

#### Results: 

<p align="center">
  <img src="images/final_video_snapshot.png" width="500" alt="Final output video">
  <br>
  <a href="images/output_video.mp4">Play final output video</a>
  <br>
</p>

#### Files Structure

```text
Football-Analysis/
├── main.py                         # End-to-end video analysis pipeline
├── yolo_inference.py               # Simple YOLO inference script
├── trackers/                       # YOLO detection, ByteTrack tracking, annotations
├── team_assigner/                  # Team assignment using K-Means on shirt colors
├── player_ball_assigner/           # Ball possession assignment by nearest player
├── camera_movement_estimator/      # Optical-flow camera movement compensation
├── view_transformer/               # Perspective transformation utilities
├── speed_and_distance_estimator/   # Player speed and distance annotation
├── utils/                          # Video and bounding-box helper functions
├── input_videos/                   # Sample input video used by main.py
├── output_videos/                  # Generated annotated video output
├── models/                         # Local YOLO model weights
├── stubs/                          # Cached tracking and camera-movement data
└── notebooks/                      # Training and team-color exploration notebooks
```

## Requirements

Required packages are listed in requirements.txt. The following code snippet checks whether the virtual environment exists. If it does not exist, it creates and activates the environment, then installs the required packages.

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$BASE_DIR/.venv"
REQUIREMENTS="$BASE_DIR/requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REQUIREMENTS"
```

## How to Run

Run the main pipeline from the project root:
```bash
cd Football-Analysis
python main.py
```

The script expects these paths to exist: `input_videos/08fd33_4.mp4` and `models/yolo5_trained/best.pt` and it writes the final annotated video to `output_videos/output_video.mp4`. The script is currently configured for the included sample video. To process another video or model, update the hardcoded paths in `main.py`.

## Notebooks

- `notebooks/training/football-training-yolo-v5.ipynb` documents YOLO training experiments on the Roboflow football dataset.
- `notebooks/team_assigning/color_assignement.ipynb` explores extracting jersey colors from cropped player images with K-Means clustering.

## Notes and Limitations

- Several paths are hardcoded in `main.py`.
- Camera movement and tracking data can be loaded from cached pickle files in `stubs/`.
- The perspective-transform source and target coordinates are hardcoded for the included camera view.
- Ball possession is assigned with a nearest-player distance heuristic, not with a physical contact model.
- Speed and distance estimates depend on the perspective transform and frame-rate assumptions in the code.

## Acknowledgements:

Special thanks to [DFL-Bundesliga-Data-Shootout-Analysis](https://github.com/aaditya29/DFL-Bundesliga-Data-Shootout-Analysi) for providing insights and well-documented resources used in this project.

© Ashkan M.,  
Released under the MIT License