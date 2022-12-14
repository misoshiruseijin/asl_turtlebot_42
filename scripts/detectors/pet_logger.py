#!/usr/bin/env python3

import rospy
from asl_turtlebot.msg import DetectedObject
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
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

        rospy.Subscriber(
            "/detector/kite", DetectedObject, self.pet_detection_callback
        )

        # Camera parameters callback
        rospy.Subscriber(
            "/camera/camera_info", CameraInfo, self.camera_info_callback
        )
        
        self.listener = tf.TransformListener()

        # Woof/Meow Publisher
        self.sound_pub = rospy.Publisher("/pet_found", String, queue_size=10)

        self.pets_detected_database = {}

        self.marker_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
        #rate = rospy.Rate(1)
        
        self.uid = {'birdgreen':1, 'dogblack':2, 'catwhite':3, 'dogwhite':4, 'catorange':5}

    def pet_detection_callback(self, msg):
        # Publish meow/woof/chirp
        dist_threshold = 0.6
        pet_class = msg.name
        sound = None
        if pet_class == "bird":
            sound = "chirp chirp" 
        elif pet_class == "dog":
            sound  = "woof"
        elif pet_class == "cat":
            sound = "meow"
        elif pet_class == 'kite':
            sound = 'whoosh'
        #if sound:
         #   self.sound_pub.publish(sound + msg.color)

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
        pet_location_world_frame = (rot_mat @ pet_location_camera_frame.T)[:3,0] + trans
        # TODO: Handle repeat detections of the same object
        # TODO: Color thresholding to get color of the pet
        x, y, z = pet_location_world_frame
        x_limit = 3.55 
        y_limit = 2.95 

        ky = pet_class+msg.color

        if ky in self.uid and msg.color!='undetermined' and msg.distance < dist_threshold and x > 0 and x < x_limit and y > 0 and y < y_limit: 
            if pet_class != "kite": 
                self.marker_publisher(trans, self.uid[ky])
            if sound:
               self.sound_pub.publish(sound + msg.color)

            if not pet_class in self.pets_detected_database:
                self.pets_detected_database[pet_class] = {}
            #if not msg.color in self.pets_detected_database[pet_class]:
             #   self.pets_detected_database[pet_class][msg.color] = ''
            
            self.pets_detected_database[pet_class][msg.color]= trans#pet_location_world_frame
            print (self.pets_detected_database)


    
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

    def marker_publisher(self, wf, kid):
        
        x, y, z = wf 
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = kid

        #self.uid +=1
        marker.type = 2 # sphere

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        self.marker_pub.publish(marker)
        print('Published marker!')
        

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    logger = PetLogger()
    logger.run()