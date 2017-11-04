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

        rospy.spin()

    def closest_wp(self, current_pose):
        """Measure the closest waypoint ahead of the car"""
        closest_idx = self.base_waypoints_tree.find_closest(util.pose_to_point(current_pose)).label
        return (closest_idx + 1) % len(self.base_waypoints)

    def pose_cb(self, msg):
        # rospy.logwarn("pose_cb")
        closest_idx = self.closest_wp(msg.pose)
        final_wp = Lane()
        final_wp.header.frame_id = msg.header.frame_id
        final_wp.header.stamp = rospy.get_rostime()
        final_wp.waypoints = util.circular_slice(self.base_waypoints, closest_idx, LOOKAHEAD_WPS)
        for i in range(LOOKAHEAD_WPS):
            self.set_waypoint_velocity(final_wp.waypoints, i, 10.0 * ONE_MPH)

        rospy.logwarn("WaypointUpdater cwi %d", closest_idx)
        self.final_waypoints_pub.publish(final_wp)

    def waypoints_cb(self, waypoints):
        # rospy.logwarn("waypoints_cb")
        self.base_waypoints = waypoints.waypoints
        self.base_waypoints_tree = util.waypoints_to_twod_tree(waypoints.waypoints)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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
