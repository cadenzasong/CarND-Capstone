"""
You can put any short functions into this file
"""

import math

def point_dist(p, q):
    """
    p: (x, y)
    q: (x, y)
    """
    return math.sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]))


def wp_to_point(wp):
    """
    wp: waypoint
    returns: (x, y)
    """
    return (wp.pose.pose.position.x, wp.pose.pose.position.y)

