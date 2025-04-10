cmake_minimum_required(VERSION 2.8.3)
project(lidar_processing_msgs)

find_package(catkin REQUIRED
  COMPONENTS
    message_generation
    std_msgs
    sensor_msgs
    pcl_ros
    nav_msgs
)

add_message_files(DIRECTORY msg
  FILES
  LidarPointCloud.msg
  lidarTrack.msg
  lidarTrackArray.msg
  GMFTTrack.msg
  GMFTTrackArray.msg
  GMFTtoSTOM.msg
  GMFTtoSTOMArray.msg
  StaticIndArray.msg
  PfGMFATrack.msg
  PfGMFATrackArray.msg
  LidarObject.msg
  LidarPerceptionOutput.msg
  LidarRoadBoundary.msg
  LaneCloudArray.msg
  RadarTracklet.msg
  RadarPerceptionOutput.msg

)

generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
    nav_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs 
    pcl_ros
    nav_msgs
)
