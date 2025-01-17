#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from planners import AStar, compute_smoothed_traj
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig
from asl_turtlebot.msg import DetectedObject, GoalPositionObject, GoalPositionList


import pdb

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    STOP = 4


class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("turtlebot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.2  # maximum velocity
        self.om_max = 0.4  # maximum angular velocity

        self.v_des = 0.12  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.1
        self.spline_deg = 3  # cubic spline
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        self.base_coords = np.array([3.2324503360558694, 1.4722777741314228, 1.5632348543459333])
        # hard-coded waypoints
        self.waypoints = [
            np.array([3.324131550403274, 2.825011104921768, 3.1383309969979205]),
            np.array([0.6401589253684375, 2.7007479803532477, -1.6930463488479157]),
            np.array([0.34697694194206596, 2.2791074670025457, -1.595382009148386]),
            np.array([0.2611280218160322, 0.38254013952875754, -0.02430574067156786]),
            np.array([2.322336742730171, 0.29914906861179397, 1.5236713646562456]),
            np.array([2.5438807940029933, 0.3091050251251528, 0.038691024881963715]),
            np.array([2.3382793584989243, 1.874743642140499, -2.6505608791010444]),
            np.array([3.301208403115228, 0.3403876021898541, -0.01759346365549529]),
            np.array([3.2324503360558694, 1.4722777741314228, 1.5632348543459333])
        ]

        # self.waypoints = [
            # np.array([3.324131550403274, 2.825011104921768, 3.1383309969979205]), # point 1
            #np.array([3.2777429548717687, 2.791419122139724, 1.4653165150734044]), #kite
            # np.array([1.8051702718271325, 2.7065318220170336, -2.855787138126887]), #black dog
            # [1.17738059 2.48085251 2.02133499]
            # x: 1.5451539952183935
            #   y: 2.586539359851155
            #   z: -0.0010121832266433602
            # np.array([0.6401589253684375, 2.7007479803532477, -1.6930463488479157]), # point 2
            # np.array([0.2667687828702921, 1.2734318776885905, -1.9385633751791418]), # blue bird
            # np.array([0.34697694194206596, 2.2791074670025457, -1.595382009148386]), # point 3
            # np.array([1.4682013423730138, 0.22978137020248612, -0.11687486366546275]), # white dog
            # np.array([0.2611280218160322, 0.38254013952875754, -0.02430574067156786]), # point 4
            # np.array([2.322336742730171, 0.29914906861179397, 1.5236713646562456]), # point 5
            # np.array([2.5438807940029933, 0.3091050251251528, 0.038691024881963715]), # point 6
            # np.array([3.301208403115228, 0.3403876021898541, -0.01759346365549529]), # point 7
            # np.array([3.2324503360558694, 1.4722777741314228, 1.5632348543459333]) # point 8
        # ]
        self.waypoint_idx = 0

        # heading controller parameters
        self.kp_th = 2.0

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.0, 0.0, 0.0, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.nav_smoothed_path_rej_pub = rospy.Publisher(
            "/cmd_smoothed_path_rejected", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        #Kite Stopping: 
        rospy.Subscriber('/detector/kite', DetectedObject, self.kite_detected_callback)
        self.stop_min_dist = 1. 
        self.stop_time = 2. 

        # rescue position
        rospy.Subscriber('/rescue_locations', GoalPositionList, self.rescue_set_callback)
        self.rescue_coords = []

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)

        print("finished init")

    def rescue_set_callback(self, msg):
        """ callback for when the detector has found a kite. Note that
        a distance of 0 can mean that the lidar did not pickup the kite at all """
        print("Received msg")
        self.rescue_coords = msg.goal_positions
        # self.rescue_coords.append(self.base_coords)

        # init_goal = self.rescue_coords[0]
        # self.x_g = init_goal.goal_pos[0]
        # self.y_g = init_goal.goal_pos[1]
        # self.theta_g = 0.0
        # self.replan()
        # goal_idx = 0

        # while goal_idx != len(self.rescue_coords): # While all pets have not been rescued
        #     print("------Current goal--------\n", self.rescue_coords[goal_idx])
        #     if self.at_goal(): # if you reached the current goal, rescue next pet
        #         goal_idx += 1
        #         self.x_g = self.rescue_coords[goal_idx].goal_pos[0]
        #         self.y_g = self.rescue_coords[goal_idx].goal_pos[1]
        #         self.theta_g = 0.0
        #         self.replan()
                

    def kite_detected_callback(self, msg):
       """ callback for when the detector has found a kite. Note that
       a distance of 0 can mean that the lidar did not pickup the kite at all """
 
       # distance of the kite
       dist = msg.distance
       #print("Kite callback entered")
 
       # if close enough and in nav mode, stop
       # and dist < self.stop_min_dist 
       # and self.mode == Mode.TRACK
       if dist > 0:
           self.init_stop()


    def init_stop(self):
       """ initiates a stop maneuver """
       self.stop_sign_start = rospy.get_rostime()
       self.mode = Mode.STOP
       #print("Switched to MODE STOP")

    def stay_idle(self):
       """ sends zero velocity to stay put """
       vel_g_msg = Twist()
       self.nav_vel_pub.publish(vel_g_msg)

    def has_stopped(self):
        """ checks if stop maneuver is over """
        return self.mode == Mode.STOP and \
            rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

        
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            rospy.logdebug(f"New command nav received:\n{data}")
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                7, #used to be 7
                self.map_probs,
            )
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
        )

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        val1 = linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
        # print(f"Value {val1} should be less than {self.at_thresh}")
        val2 = abs(wrapToPi(self.theta - self.theta_g))
        # print(f"Value for theta {val2} should be less than {self.at_thresh_theta}")
        return (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
        )

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(
                self.x, self.y, self.theta, t
            )
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """

        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        t_new, traj_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_deg, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                self.publish_smoothed_path(
                    traj_new, self.nav_smoothed_path_rej_pub
                )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        import time
        # time.sleep(40)
        # self.gen_map() # exploration
    
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            #if self.mode == Mode.EXPLORE:
                # time.sleep(40)
                # pdb.set_trace()
                # self.gen_map()

             #   thresh = 0.01
                #for point in self.waypoints:
                 #   print (f'!!!!!!!!!!!!!!!!!!!!!??{point[0]}')
            print("STATE:", self.mode)
            if self.mode == Mode.IDLE:
                if (self.waypoints):
                    self.x_g, self.y_g, self.theta_g = self.waypoints[0]
                    self.replan()
                elif (self.rescue_coords):
                    self.x_g, self.y_g, self.theta_g = self.rescue_coords[0].goal_pos
                    self.replan()
                # pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.STOP:
                print("Entered MODE STOP")
                self.stay_idle()
                if self.has_stopped():
                    print("Leaving MODE STOP, starting replan")
                    if self.x_g is not None and self.y_g is not None:
                        self.replan() 
                    else:
                        self.switch_mode(Mode.IDLE)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.x_g is not None and self.y_g is not None:
                    if self.at_goal():
                        # forget about goal:
                        self.x_g = None
                        self.y_g = None
                        self.theta_g = None
                        self.switch_mode(Mode.IDLE)
                        if self.waypoints:
                            self.waypoints.pop(0)
                            print(f"\nWAYPOINTS: {self.waypoints}")
                        elif self.rescue_coords:
                            self.rescue_coords.pop(0)
                            print(f"\nRescue Coordinates: {self.rescue_coords}")
                            self.init_stop()
                else:
                    self.switch_mode(Mode.IDLE)

            # print(f"!!!!! {self.x_g}, {self.y_g}, {self.theta_g} !!!!!")
            self.publish_control()
            rate.sleep()


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    print("MODE IS:")
    print(nav.mode)
    nav.run()