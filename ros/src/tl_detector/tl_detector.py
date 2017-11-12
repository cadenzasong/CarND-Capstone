#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

import common_helpers.twod_tree as twod_tree
import common_helpers.util as util
import os
import os.path
import time
import math

STATE_COUNT_THRESHOLD = 3
# Approximate average distance between two waypoints in meter
WAYPOINT_DIST = 0.3
# Maximal look ahead for traffic lights in meter
LOOKAHEAD_DIST = 100
CAMERA_HEIGHT = 1.0
CAMERA_ANGLE_MAX = 0.225
CAMERA_ANGLE_MIN = 0.0214
TRAFFIC_LIGHT_VERT_OFFSET = 0.39
TRAFFIC_LIGHT_HEIGHT = 1.77
TRAFFIC_LIGHT_WIDTH = 0.78

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.yaw = None
        self.waypoints = None
        self.waypoint_tree = None
        self.camera_image = None
        self.cropped_image = None
        self.lights = []
        self.lights_wpi = [] # Waypoint indices for traffix lights

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_wpi = []  # Waypoint indices for stop lines

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        _, _, self.yaw = tf.transformations.euler_from_quaternion([msg.pose.orientation.x,
                              msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        #rospy.loginfo("Car position %s %s, yaw angle %s", msg.pose.position.x, msg.pose.position.y, self.yaw)

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.waypoint_tree = util.waypoints_to_twod_tree(waypoints.waypoints)


    def traffic_cb(self, msg):
        self.lights = msg.lights

        if not self.waypoint_tree:
            return

        if self.lights_wpi and self.stop_line_wpi:
            return

        for light in self.lights:
            x = light.pose.pose.position.x
            y = light.pose.pose.position.y
            self.lights_wpi.append(self.waypoint_tree.find_closest((x,y)).label)

        for pos in self.stop_line_positions:
            self.stop_line_wpi.append(self.waypoint_tree.find_closest((pos[0], pos[1])).label)


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        rospy.loginfo("Light %s %s", light_wp, state)

        if light_wp != -1:
            self.save_training_image(state, self.cropped_image)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def save_training_image(self, state, image):
        image_dir = './train_img/%d' % state
        #image_dir = '/home/student/sim-images'
        time_str = ("%.3f" % time.time()).replace('.','_')
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        file_name = image_dir + '/'+time_str+'.jpg'
        cv2.imwrite(file_name, image)
        rospy.loginfo("writing image: " + file_name)

    def crop_region_of_interest(self, light):

        #fx = self.config['camera_info']['focal_length_x']
        #fy = self.config['camera_info']['focal_length_y']
        # Not given for the simulator
        fx = 2888
        fy = 2888
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # Convert traffic light position into vehice coordinates
        dx = light.pose.pose.position.x - self.pose.pose.position.x
        dy = light.pose.pose.position.y - self.pose.pose.position.y

        tl_s = dx * math.cos(-self.yaw) - dy * math.sin(-self.yaw)
        tl_d = dx * math.sin(-self.yaw) + dy * math.cos(-self.yaw)
        tl_h = light.pose.pose.position.z

        rospy.loginfo("Light position %s %s => %s %s, height %s",
            light.pose.pose.position.x, light.pose.pose.position.y, tl_s, tl_d, tl_h)

        # Calculate position and size in image using camera focal length
        # Assume that camera height is 1 m and that camera is looking up a bit
        roi_x = int(image_width / 2 - (tl_d + TRAFFIC_LIGHT_VERT_OFFSET) / tl_s * fx)
        roi_y = int(image_height - (tl_h - CAMERA_HEIGHT - tl_s * math.sin(CAMERA_ANGLE_MIN)) / tl_s * fy)
        roi_width = int(TRAFFIC_LIGHT_WIDTH / tl_s * fx)
        roi_height = int(TRAFFIC_LIGHT_HEIGHT / tl_s * fy)

        rospy.loginfo("Region of interest center x %s, center y %s, width %s height %s",
            roi_x, roi_y, roi_width, roi_height)

        # Crop image to region of interest
        x1 = int(roi_x - roi_width / 2)
        x2 = int(roi_x + roi_width / 2)
        y1 = int(roi_y - roi_height / 2)
        y2 = int(roi_y + roi_height / 2)

        rospy.loginfo("Crop x1 %s, x2 %s, y1 %s y2 %s", x1, x2, y1, y2)

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255,0,0), 3)
        self.cropped_image = cv_image
        #self.cropped_image = cv_image[y1:y2, x1:x2]

    def wp_dist(self, ahead, astern):
        """Get the distance between two waypoint indices, handle overflow

        Args:
            ahead: position of waypoint ahead
            astern: position of waypoint astern (behind)

        Returns:
            int: distance of given waypoint indices

        """
        if ahead > astern:
            return ahead - astern
        return ahead + len(self.waypoints.waypoints) - astern

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        x = pose.position.x
        y = pose.position.y
        closest_waypoint_index = self.waypoint_tree.find_closest((x,y)).label
        #rospy.loginfo("TLDetector cwi %d", closest_waypoint_index)
        return closest_waypoint_index


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # if some data is not ready yet, just return
        if not (self.pose and self.waypoints and self.lights):
            return -1, TrafficLight.UNKNOWN

        light = None
        car_position = self.get_closest_waypoint(self.pose.pose)

        min_dist = 1e42
        light = None
        light_wp = -1

        # TODO: Maybe we will need to find the first light whose _stop_line_ is ahead the car
        # Iterate over all traffic light waypoints
        for index, wp in enumerate(self.lights_wpi):
            act_dist = self.wp_dist(wp, car_position)
            # If traffic light is ahead of car and in visual range
            if (act_dist > 0) and (act_dist < LOOKAHEAD_DIST/WAYPOINT_DIST):
                if act_dist < min_dist:
                    min_dist = act_dist
                    light_wp = wp
                    light = self.lights[index]

        if light:
            min_dist = 1e42
            stop_line_wp = -1
            # Iterate over all stop lane waypoints
            for index, wp in enumerate(self.stop_line_wpi):
                act_dist = self.wp_dist(light_wp, wp)
                # If traffic light is ahead of stop line and in visual range
                if (act_dist > 0) and (act_dist < LOOKAHEAD_DIST/WAYPOINT_DIST):
                    if act_dist < min_dist:
                        min_dist = act_dist
                        stop_line_wp = wp

            if stop_line_wp:
                self.crop_region_of_interest(light)
                #TODO state = self.get_light_state(light)
                #return stop_line_wp, state
                return stop_line_wp, light.state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
