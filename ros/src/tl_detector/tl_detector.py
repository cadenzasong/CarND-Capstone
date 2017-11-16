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
from scipy.misc import imsave
import common_helpers.twod_tree as twod_tree
import common_helpers.util as util
import os
import os.path
import time

STATE_COUNT_THRESHOLD = 3
# Approximate average distance between two waypoints in meter
WAYPOINT_DIST = 0.3
# Maximal look ahead for traffic lights in meter
LOOKAHEAD_DIST = 150

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoint_tree = None
        self.camera_image = None
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
        self.light_classifier = None
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.image_under_process = 0
        self.has_image = False
        
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            #rospy.loginfo('Entering infinite loop')
            
            if not self.waypoint_tree:
                continue
            if not self.has_image:
                continue
            
            light_wp, state = self.process_traffic_lights()

            rospy.loginfo("Light %s %s", light_wp, state)
    
            #self.save_training_image(state)
    
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
            
            rate.sleep()
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.waypoint_tree = util.waypoints_to_twod_tree(waypoints.waypoints)


    def traffic_cb(self, msg):
        self.lights = msg.lights

        if not self.waypoint_tree:
            return

        # construct self.lights_wpi and self.stop_line_wpi the first time
        # 8 lights_wpi and 8 stop_line_wpi (ahead of lights)
        #       light   stop_line
        # [0]:  318     292
        # [1]:  784     753
        # [2]:  2095    2047
        # [3]:  2625    2580
        # [4]:  6322    6294
        # [5]:  7036    7008
        # [6]:  8565    8540
        # [7]:  9773    9733
        if not (self.lights_wpi and self.stop_line_wpi):
            for light in self.lights:
                x = light.pose.pose.position.x
                y = light.pose.pose.position.y
                self.lights_wpi.append(self.waypoint_tree.find_closest((x,y)).label)

            for pos in self.stop_line_positions:
                self.stop_line_wpi.append(self.waypoint_tree.find_closest((pos[0], pos[1])).label)


#==============================================================================
#         # bypass the classifier, publish the ground truth from /vehicle/traffic_lights
#         car_wpi = self.waypoint_tree.find_closest((self.pose.pose.position.x, self.pose.pose.position.y)).label
#         # rospy.loginfo('car_wpi: %d' % car_wpi)
#         startIdx = 0
#         if car_wpi < self.stop_line_wpi[-1]:
#             while car_wpi >= self.stop_line_wpi[startIdx]:
#                 startIdx += 1
# 
#         upcoming_red_light_wpi = -1 # no red light
#         for i in range(len(self.stop_line_wpi)):
#             if self.lights[startIdx % len(self.lights)].state == TrafficLight.RED:
#                 upcoming_red_light_wpi = self.stop_line_wpi[startIdx % len(self.lights)]
#                 break;
#         # rospy.loginfo('upcoming_red_light_wpi: %d' % upcoming_red_light_wpi)
#         self.upcoming_red_light_pub.publish(Int32(upcoming_red_light_wpi))
#==============================================================================
    
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        
        #rospy.loginfo("Light")
        
        if not self.waypoint_tree:
            return
        self.has_image = True
        self.camera_image = msg


    def save_training_image(self, state):
        image_dir = '/home/student/sim-images/%d' % state
        #image_dir = '/home/student/sim-images'
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        time_str = ("%.3f" % time.time()).replace('.','_')
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        file_name = image_dir + '/'+time_str+'.jpg'
        cv2.imwrite(file_name, cv_image)
        rospy.loginfo("writing image: " + file_name)

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
        # rospy.loginfo("TLDetector cwi %d", closest_waypoint_index)
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
        test_im = self.camera_image
        cv_image = self.bridge.imgmsg_to_cv2(test_im, "rgb8")
        light_color = TrafficLight.UNKNOWN
        light_color = self.light_classifier.get_classification(cv_image)
        #Get classification
#        if(self.image_under_process==0):
#            self.image_under_process=1
#            light_color = self.light_classifier.get_classification(cv_image)
#            self.image_under_process=0
        return light_color

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
        #tmp_ph = self.get_light_state(light)
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
                state = self.get_light_state(light)
                return stop_line_wp, state
                #return stop_line_wp, light.state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
