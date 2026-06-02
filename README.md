# DFL - Bundesliga Data Shootout Analysis using YOLO, OpenCV, and Python

[Kaggle](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview)

As the initial inference of the YOLO model for detecting players and the ball on the pitch was not satisfactory, we fine-tuned YOLOv5 on a football dataset from Roboflow to detect players, referees, goalkeepers, and the ball. The trained model now excludes people outside the pitch, and the ball is detected more accurately.

Now, we want to make the annotations on the predicted video a bit neater, with information that is easier to follow.