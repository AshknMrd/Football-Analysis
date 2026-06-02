# DFL - Bundesliga Data Shootout Analysis using YOLO, OpenCV, and Python

[Kaggle](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview)

As the initial inference of the YOLO model for detecting players and the ball on the pitch was not satisfactory, we fine-tuned YOLOv5 on a football dataset from Roboflow to detect players, referees, goalkeepers, and the ball. The trained model now excludes people outside the pitch, and the ball is detected more accurately.


Now is also the time to prepare the Tracker class that can tracks different objects througout the entrire video and all frames.

I many instance the goal keeper is also detected as a player and switches between these two classes. This might be du to the small training dataset and as we do not investigating any statistics on the goal keeper here we treat the goalkeeper as a layer also. 


Now, we want to make the annotations on the predicted video a bit neater, with information that is easier to follow. In the tracker.py file: 
- Draw a red elipse under each player proportionate with its bounding box width
- Draw a yellow elipse under the referees
- Put the players id (detected object id) also printed under the player
- Put a pointer on the ball

At this step, we want to seperate the players as team one and two based on the color of their shirts.
- Crop the image of each player using the bounding box.
- Tekae the top half of the image as always includes the T-shirt that seperates the teams by color.
- Now cluster the top half into two class of beckground and T-shirt color (using K-mean for the clustering)
- Take the color of the segmented player using K-mean of the center for all players at each frame
- Divide the all players color into only 2 teams again using K-means.
- A team id is assigned to each player and if theat is already decided for a player (by checking the player id) that team assigning would not be runned in the next frame.

Now, since the ball is not detected in every frames and the fact that ball move in a stright line, we will fill the missing frames to detect the ball with average location of the 2 known ends. 
