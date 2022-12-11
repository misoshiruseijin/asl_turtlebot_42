import rospy
from asl_turtlebot.msg import DetectedObject
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

class PetLogger:

    def __init__(self):
        rospy.init_node("pet_logger", anonymous=True)
        self.cx, self.cy, self.fx, self.fy = 0, 0, 0, 0

        # Subscribe to mobilenet detector output
        self.cat_sub = rospy.Subscriber(
            "/detector/cat", DetectedObject, self.pet_detection_callback
        )

        # Camera parameters callback
        self.camera_info_sub = rospy.Subscriber(
            "/camera/camera_info", CameraInfo, self.camera_info_callback
        )

        # Woof/Meow Publisher
        self.sound_pub = rospy.Publisher("/pet_found", String, queue_size=10)

        self.pets_detected_database = {}
        

    def pet_detection_callback(self, msg):
        # Publish meow/woof
        pet_class = msg.name
        sound = "meow" if pet_class == "cat" else "woof"
        self.sound_pub.pub(sound)

        # Store location
        box = msg.corners
        box_center_x, box_center_y = box[3] - box[1], box[2] - box[0]
        pet_location_camera_frame = self.project_pixel_to_world(
            box_center_x, 
            box_center_y, 
            msg.distance
        )
        #TODO: Use transform tree to convert from camera frame to world frame
        pet_location_world_frame = ()

        if not pet_class in self.pets_detected_database:
            self.pets_detected_database[pet_class] = {}
        if not msg.color in self.pets_detected_database[pet_class]:
            self.pets_detected_database[pet_class][msg.color] = []
        self.pets_detected_database[pet_class][msg.color].append(pet_location_world_frame)

    
    def project_pixel_to_world(self, u, v, dist):
        """takes in a pixel coordinate (u,v) and returns a tuple (x,y,z)
        that is a unit vector in the direction of the pixel, in the camera frame.
        This function access self.fx, self.fy, self.cx and self.cy"""

        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return (x*dist, y*dist, dist)

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

if __name__ == "__main___":
    logger = PetLogger()
    logger.run()