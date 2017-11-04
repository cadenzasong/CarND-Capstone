"""
You can put any short functions into this file
"""

import math
import twod_tree

def point_dist(p, q):
    """
    p: (x, y)
    q: (x, y)
    """
    return math.sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]))

def pose_to_point(pose):
    """
    pose: pose
    returns: (x, y)
    """
    return (pose.position.x, pose.position.y)

def wp_to_point(wp):
    """
    wp: waypoint
    returns: (x, y)
    """
    return (wp.pose.pose.position.x, wp.pose.pose.position.y)

def waypoints_to_twod_tree(waypoints):
    items = []
    for wp_index, wp in enumerate(waypoints):
        items.append(twod_tree.LabeledPoint(wp_index, wp_to_point(wp)))

    return twod_tree.TwoDTree(items)

def circular_slice(arr, from_, n):
    """
    arr: array
    from_: starting index
    n: how many elements you want
    """
    assert from_ >= 0 and from_ < len(arr)
    res = []
    while n > 0:
        k = min(n, len(arr) - from_)
        res += arr[from_:from_ + n]
        n -= k
        from_ = 0
    return res
