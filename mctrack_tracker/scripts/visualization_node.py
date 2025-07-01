#!/usr/bin/env python3
import math
import traceback
from typing import List, Dict

import numpy as np
from pyquaternion import Quaternion

import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from lidar_processing_msgs.msg import PfGMFATrack, PfGMFATrackArray

class MCTrackVisualizerNode:
    def __init__(self):
        rospy.init_node("mctrack_visualizer_node", anonymous=True)

        self.vis_pub = rospy.Publisher("/tracking/markers", MarkerArray, queue_size=10)

        self.tracks_sub = rospy.Subscriber("/tracking/objects",
                                           PfGMFATrackArray,
                                           self.tracks_callback,
                                           queue_size=100,
                                           tcp_nodelay=True)
    
        self.marker_array = MarkerArray()
        self.prev_track_ids = set()

        rospy.loginfo("MCTrackVisualizerNode initialized.")

    def tracks_callback(self, msg: PfGMFATrackArray):
        try:
            tracks = msg.tracks
            vis_header = Header(frame_id="vehicle", stamp=msg.header.stamp)

            current_ids = set(t.id for t in tracks)
            deleted_ids = self.prev_track_ids - current_ids
            self.marker_array = MarkerArray()

            for tid in deleted_ids:
                for ns, base_id in [("track_meshes", 0), ("track_ids", 1000), ("track_arrows", 2000)]:
                    m = Marker()
                    m.header = vis_header
                    m.ns = ns
                    m.id = base_id + tid
                    m.action = Marker.DELETE
                    self.marker_array.markers.append(m)

            for t in tracks:
                m1 = self.create_single_track_marker(t, vis_header, t.id)
                self.marker_array.markers.append(m1)

                m2 = self.create_text_marker(t, vis_header, 1000 + t.id)
                if m2 is not None:
                    self.marker_array.markers.append(m2)

                m3 = self.create_arrow_marker(t, vis_header, 2000 + t.id)
                self.marker_array.markers.append(m3)

            ego_marker = self.create_ego_marker(vis_header.stamp)
            self.marker_array.markers.append(ego_marker)

            self.vis_pub.publish(self.marker_array)
            self.prev_track_ids = current_ids

        except Exception as e:
            rospy.logerr(f"[Visualizer] Error: {e}\n{traceback.format_exc()}")

    def create_single_track_marker(self, track: PfGMFATrack, header: Header, marker_id: int) -> Marker:
        m = Marker()
        m.header = header
        m.ns = "track_meshes"
        m.id = marker_id
        m.action = Marker.ADD
        m.type = Marker.MESH_RESOURCE
        m.mesh_use_embedded_materials = True

        m.pose.position.x = track.pos_x
        m.pose.position.y = track.pos_y
        m.pose.position.z = 0.0

        q = tf.transformations.quaternion_from_euler(0, 0, track.yaw)
        m.pose.orientation.x = q[0]
        m.pose.orientation.y = q[1]
        m.pose.orientation.z = q[2]
        m.pose.orientation.w = q[3]

        m.scale.x = track.boundingbox[0]
        m.scale.y = track.boundingbox[1]
        m.scale.z = track.boundingbox[2]

        m.color.a = min(track.confidence_ind * 5, 1.0)
        m.color.r = 0.0
        m.color.g = 0.2
        m.color.b = 1.0

        class_mesh_paths = {
            1: "package://vdcl_fusion_perception/marker_dae/Car.dae",
            2: "package://vdcl_fusion_perception/marker_dae/Truck.dae",
            3: "package://vdcl_fusion_perception/marker_dae/Bus.dae",
            4: "package://vdcl_fusion_perception/marker_dae/Trailer.dae",
            5: "package://vdcl_fusion_perception/marker_dae/Truck.dae",
            6: "package://vdcl_fusion_perception/marker_dae/Pedestrian.dae",
            7: "package://vdcl_fusion_perception/marker_dae/Motorcycle.dae",
            8: "package://vdcl_fusion_perception/marker_dae/Bicycle.dae",
            9: "package://vdcl_fusion_perception/marker_dae/Barrier.dae",
            10: "package://vdcl_fusion_perception/marker_dae/TrafficCone.dae",
        }
        m.mesh_resource = class_mesh_paths.get(track.obj_class, "")
        m.lifetime = rospy.Duration(0.2)
        return m

    def create_text_marker(self, track: PfGMFATrack, header: Header, marker_id: int) -> Marker:
        t_m = Marker()
        t_m.header = header
        t_m.ns = "track_ids"
        t_m.id = marker_id
        t_m.type = Marker.TEXT_VIEW_FACING
        t_m.action = Marker.ADD
        t_m.pose.position.x = track.pos_x
        t_m.pose.position.y = track.pos_y
        t_m.pose.position.z = track.boundingbox[2] + 1.0
        t_m.scale.z = 0.8
        t_m.color.a = 1.0
        t_m.color.r = 1.0
        t_m.color.g = 1.0
        t_m.color.b = 1.0
        t_m.text = str(track.id)
        t_m.lifetime = rospy.Duration(0.2)
        return t_m

    def create_arrow_marker(self, track: PfGMFATrack, header: Header, marker_id: int) -> Marker:
        arrow = Marker()
        arrow.header = header
        arrow.ns = "track_arrows"
        arrow.id = marker_id
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.2
        arrow.scale.y = 0.5
        arrow.scale.z = 0.3
        arrow.color.r = 1.0
        arrow.color.g = 1.0
        arrow.color.b = 1.0
        arrow.color.a = 1.0

        start = Point(x=track.pos_x, y=track.pos_y, z=track.boundingbox[2]/2.0)
        end = Point(x=track.pos_x + math.cos(track.yaw),
                    y=track.pos_y + math.sin(track.yaw),
                    z=start.z)
        arrow.points.append(start)
        arrow.points.append(end)

        arrow.lifetime = rospy.Duration(0.2)
        return arrow

    def create_ego_marker(self, stamp):
        marker = Marker()
        marker.header.frame_id = "vehicle"
        marker.header.stamp = stamp
        marker.ns = "ego_vehicle"
        marker.id = 9999
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = "package://vdcl_fusion_perception/marker_dae/Car.dae"
        marker.mesh_use_embedded_materials = True

        marker.pose.position.x = 1.5
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        quaternion = Quaternion(axis=[0, 0, 1], angle=0)
        marker.pose.orientation.w = quaternion[0]
        marker.pose.orientation.x = quaternion[1]
        marker.pose.orientation.y = quaternion[2]
        marker.pose.orientation.z = quaternion[3]

        marker.scale.x = 4.0
        marker.scale.y = 2.0
        marker.scale.z = 2.0

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.lifetime = rospy.Duration(0.2)

        return marker

if __name__ == '__main__':
    try:
        MCTrackVisualizerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
