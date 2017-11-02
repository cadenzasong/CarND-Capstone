#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

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


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose = None
        self.waypoints = None
        
        rospy.spin()
  
    def closest_wp (self,waypoints):
        #Measure the closest waypoint ahead of the car 
        min_dist = 10000
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        #Search in all waypoints
        for i in range(0, LOOKHEAD_WPS):
            if (waypoints[i].pose.pose.position.x > 0):
                #This value is in front of the car
                dist = dl(self.current_pose.position, waypoints[i].pose.pose.position)
            #Compare if the calculated distance is the closest distance
            if (dist < min_dist):
                min_dist = dist
                #Save the index of the closest point ahead of the car
                closest_indx=i
        return closest_indx
    
    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.waypoints = waypoints.waypoints
        #Check if the waypoints are received
        if not self.waypoints:
            #Get the index of the closest waypoint in the waypoints
            closest_indx = closest_wp (self.current_pose,self.waypoints) 

        #Create final_wp msg that has Lane type. Fill it and then publish it
        final_wp = Lane()
        final_wp.header.frame_id = waypoints.header.frame_id
        final_wp.header.stamp = rospy.get_rostime()
        final_wp.waypoints = self.waypoints[closest_indx:closest_indx+LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(final_wp)
        pass
    
    def pose_cb(self, msg):
        # TODO: Implement
        self.current_pose = msg
        pass

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
