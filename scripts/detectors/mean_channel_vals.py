import cv2
import numpy as np

paths = [
    "/data/home/group40/catkin_ws/src/asl_turtlebot_42/models/dog_black/materials/textures/black_dog.png",
    "/data/home/group40/catkin_ws/src/asl_turtlebot_42/models/dog_white/materials/textures/white_dog.png",
    "/data/home/group40/catkin_ws/src/asl_turtlebot_42/models/dog_orange/materials/textures/orange_dog.png",
   "/data/home/group40/catkin_ws/src/asl_turtlebot_42/models/cat_orange/materials/textures/orange_cat.png",
    "/data/home/group40/catkin_ws/src/asl_turtlebot_42/models/cat_black/materials/textures/black_cat.png",
]
names = ["black dog", "white dog", "orange dog", "orange cat", "blue bird"]
for name, p in zip(names, paths):
    im = cv2.imread(p)
    means = np.mean(im, axis=(0,1))
    print(name)
    print("BGR:", means)
    # cv2.imshow('win', im)
    # cv2.waitKey(0)
    print("-----")
