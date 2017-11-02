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
import twod_tree

STATE_COUNT_THRESHOLD = 3
# Approximate average distance between two waypoints in meter
WAYPOINT_DIST = 0.3
# Maximal look ahead for traffic lights in meter
LOOKAHEAD_DIST = 300

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_tree = None
        self.camera_image = None
        self.lights = []
        self.lights_wpi = [] # Waypoint indices for traffix lights

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

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

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        items = []
        for wp_index, wp in enumerate(self.waypoints.waypoints):
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            items.append(twod_tree.LabeledPoint(wp_index, (wp_x, wp_y)))

        self.waypoint_tree = twod_tree.TwoDTree(items)


    def traffic_cb(self, msg):
        self.lights = msg.lights

        if self.waypoints != None:
            # Use the waypoint indices as some kind of coarse s coordinate along the track
            if not self.lights_wpi:  # only do this once as traffic lights don't usually move
                for light in self.lights:
                    x = light.pose.position.x
                    y = light.pose.position.y
                    self.lights_wpi.append(self.waypoint_tree.find_closest((x,y)).label)

            if not self.stop_line_wpi:  # only do this once as stop lines don't usually move
                for pos in self.stop_line_positions:
                    x = pos.x
                    y = pos.y
                    self.stop_line_wpi.append(self.waypoint_tree.find_closest((x,y)).label)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

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

    def wp_dist(ahead, astern):
        """Get the distance between two waypoint indices, handle overflow

        Args:
            ahead: position of waypoint ahead
            astern: position of waypoint astern

        Returns:
            int: distance of given waypoint indices

        """
        if ahead > astern:
            return ahead - astern
        return len(self.waypoints) - ahead + astern

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
        light = None

        if(self.pose and self.waypoint_tree):
            car_position = self.get_closest_waypoint(self.pose.pose)

        min_dist = 1e42
        light = None
        light_wp = -1
        # Iterate over all traffic light waypoints
        for index, wp in enumerate(self.lights_wpi):
            act_dist = wp_dist(wp, car_position)
            # If traffic light is ahead of car and in visual range
            if (act_dist > 0) and (act_dist < LOOKAHEAD_DIST/WAYPOINT_DIST):
                if act_dist < min_dist:
                    min_dist = act_dist
                    light_wp = wp
                    light = self.lights[index]

        if light:
            min_dist = 1e42;
            stop_line_wp = -1
            # Iterate over all stop lane waypoints
            for index, wp in enumerate(self.stop_line_wpi):
                act_dist = wp_dist(light_wp, wp)
                # If traffic light is ahead of stop line and in visual range
                if (act_dist > 0) and (act_dist < LOOKAHEAD_DIST/WAYPOINT_DIST):
                    if act_dist < min_dist:
                        min_dist = act_dist
                        stop_line_wp = wp

            state = self.get_light_state(light)
            return stop_line_wp, state
        # I don't get it why the next line would be a good idea?
        # Neither do I - should be removed...
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
