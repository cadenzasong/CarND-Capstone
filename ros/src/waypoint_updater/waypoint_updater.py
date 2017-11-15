#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import sys

import math
import common_helpers.util as util

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704
TARGET_SPEED = 25.0 * ONE_MPH

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.base_waypoints_tree = None
        self.pose = None
        self.red_light_wp = -1

        rospy.spin()

    def closest_wp(self, current_pose):
        """Measure the closest waypoint ahead of the car"""
        closest_idx = self.base_waypoints_tree.find_closest(util.pose_to_point(current_pose)).label
        return (closest_idx + 1) % len(self.base_waypoints)

    def pose_cb(self, msg):
        # rospy.logwarn("pose_cb")
        self.pose = msg.pose
        if self.base_waypoints_tree:
            self.update_waypoints()

    def waypoints_cb(self, waypoints):
        # rospy.logwarn("waypoints_cb")
        self.base_waypoints = waypoints.waypoints
        self.base_waypoints_tree = util.waypoints_to_twod_tree(waypoints.waypoints)

    def update_waypoints(self):
        closest_idx = self.closest_wp(self.pose)
        #rospy.logwarn('update_waypoints, closest_idx = %d, self.red_light_wp = %d' % (closest_idx, self.red_light_wp))
        final_wp = Lane()
        final_wp.header.stamp = rospy.get_rostime()
        final_wp.waypoints = util.circular_slice(self.base_waypoints, closest_idx, LOOKAHEAD_WPS)
        ignore_tl = False
        if self.red_light_wp == -1:
            #rospy.logwarn('no red light, ignore')
            # ignore TL
            for i in range(LOOKAHEAD_WPS):
                self.set_waypoint_velocity(final_wp.waypoints, i, TARGET_SPEED)
        else:
            red_light_wp = self.red_light_wp
            if red_light_wp < closest_idx:
                red_light_wp += len(self.base_waypoints)
            if (red_light_wp - closest_idx) > 125:
                # ignore TL
                #rospy.logwarn('red light is too far ahead, ignore')
                for i in range(LOOKAHEAD_WPS):
                    self.set_waypoint_velocity(final_wp.waypoints, i, TARGET_SPEED)
            else:
                zero_velocity_idx = red_light_wp - closest_idx
                current_velocity = min(TARGET_SPEED, self.get_waypoint_velocity(final_wp.waypoints[0]))
                #rospy.logwarn('zero_velocity_idx = %d' % (zero_velocity_idx))
                for i in range(LOOKAHEAD_WPS):
                    target_velocity = 0.0
                    if i < zero_velocity_idx and zero_velocity_idx - i >= 12:
                        scale = 1.0 * (zero_velocity_idx - i) / zero_velocity_idx
                        target_velocity = current_velocity * scale
                        #if i < 10:
                        #    rospy.logwarn('[%d] target velocity = %f' % (i, target_velocity))
                    self.set_waypoint_velocity(final_wp.waypoints, i, target_velocity)
        self.final_waypoints_pub.publish(final_wp)

    def traffic_cb(self, msg):
        self.red_light_wp = msg.data
        if self.base_waypoints_tree:
            self.update_waypoints()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
