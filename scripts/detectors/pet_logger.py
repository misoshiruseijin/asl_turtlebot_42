#!/usr/bin/env python3

import rospy
from asl_turtlebot.msg import DetectedObject
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
import tf
from tf.transformations import quaternion_matrix
import numpy as np

class PetLogger:

    def __init__(self):
        rospy.init_node("pet_logger", anonymous=True)
        self.cx, self.cy, self.fx, self.fy = 0, 0, 0, 0
        
        # Subscribe to mobilenet detector output
        rospy.Subscriber(
            "/detector/dog", DetectedObject, self.pet_detection_callback
        )

        rospy.Subscriber(
            "/detector/bird", DetectedObject, self.pet_detection_callback
        )

        rospy.Subscriber(
            "/detector/cat", DetectedObject, self.pet_detection_callback
        )

        # Camera parameters callback
        rospy.Subscriber(
            "/camera/camera_info", CameraInfo, self.camera_info_callback
        )
        
        self.listener = tf.TransformListener()

        # Woof/Meow Publisher
        self.sound_pub = rospy.Publisher("/pet_found", String, queue_size=10)

        self.pets_detected_database = {}

    def pet_detection_callback(self, msg):
        # Publish meow/woof/chirp
        pet_class = msg.name
        sound = None
        if pet_class == "bird":
            sound = "chirp chirp" 
        elif pet_class == "dog":
            sound  = "woof"
        elif pet_class == "cat":
            sound = "meow"
        if sound:
            self.sound_pub.publish(sound)

        # Store location
        box = msg.corners # [ymin, xmin, ymax, xmax] 
        box_center_x, box_center_y = box[3] - box[1], box[2] - box[0]
        pet_location_camera_frame = self.project_pixel_to_world(
            box_center_x, 
            box_center_y, 
            msg.distance
        )
        #TODO: Use transform tree to convert from camera frame to world frame
        try:
            (trans,rot) = self.listener.lookupTransform('/odom', '/base_camera', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        rot_mat = quaternion_matrix(rot)
        print("R:", rot_mat)
        print("trans:", trans)
        pet_location_world_frame = (rot_mat @ pet_location_camera_frame.T)[:3,0] + trans
        print("PET LOCATION WORLD ESTIMATE:", pet_location_world_frame)
        # pet_location_world_frame = ()

        # if not pet_class in self.pets_detected_database:
        #     self.pets_detected_database[pet_class] = {}
        # if not msg.color in self.pets_detected_database[pet_class]:
        #     self.pets_detected_database[pet_class][msg.color] = []
        # self.pets_detected_database[pet_class][msg.color].append(pet_location_world_frame)

    
    def project_pixel_to_world(self, u, v, dist):
        """takes in a pixel coordinate (u,v) and returns a tuple (x,y,z)
        that is a unit vector in the direction of the pixel, in the camera frame.
        This function access self.fx, self.fy, self.cx and self.cy"""

        x = (u - self.cx) / self.fx # -v
        y = (v - self.cy) / self.fy # -u
        z = 1.0 # depth-> x

        # Convert to unit vector
        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= norm
        y /= norm
        z /= norm

        scale = dist / z
        y_camera = y*scale
        x_camera = x*scale
        z_camera = z*scale
        
        return np.array([x_camera, y_camera, z_camera, 1.0]).reshape((1,4)) # camera frame

    def camera_info_callback(self, msg):
        """extracts relevant camera intrinsic parameters from the camera_info message.
        cx, cy are the center of the image in pixel (the principal point), fx and fy are
        the focal lengths. Stores the result in the class itself as self.cx, self.cy,
        self.fx and self.fy"""

        if any(msg.P):
            self.cx = msg.P[2]
            self.cy = msg.P[6]
            self.fx = msg.P[0]
            self.fy = msg.P[5]
        else:
            rospy.loginfo("`CameraInfo` message seems to be invalid; ignoring it.")

    def pet_query(self, pet_class, color):
        try:
            return self.self.pets_detected_database[pet_class][color]
        except KeyError:
            print("No match for pet query found")
            return None 

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    logger = PetLogger()
    logger.run()