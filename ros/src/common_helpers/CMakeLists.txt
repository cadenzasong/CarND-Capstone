cmake_minimum_required(VERSION 2.8.3)
project(common_helpers)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  waypoint_updater
)

catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
