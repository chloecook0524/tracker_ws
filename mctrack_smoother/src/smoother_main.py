#!/usr/bin/env python3
import sys
import rospy
import gtsam
import tf.transformations
from std_msgs.msg import Header
from lidar_processing_msgs.msg import PfGMFATrack, PfGMFATrackArray
from visualization_msgs.msg import Marker, MarkerArray
from bag_parser import load_tracks_from_bag
from factor_graph_builder import build_graph_from_track

def create_marker(track, pose, marker_id, header):
    m = Marker()
    m.header = header
    m.ns = "smoothed"
    m.id = marker_id
    m.type = Marker.CUBE
    m.action = Marker.ADD

    m.pose.position.x = pose.x()
    m.pose.position.y = pose.y()
    m.pose.position.z = 0.5

    q = tf.transformations.quaternion_from_euler(0, 0, pose.theta())
    m.pose.orientation.x = q[0]
    m.pose.orientation.y = q[1]
    m.pose.orientation.z = q[2]
    m.pose.orientation.w = q[3]

    l, w, h = track['size'][:3]
    m.scale.x = l
    m.scale.y = w
    m.scale.z = h if h > 0 else 1.5

    m.color.a = 1.0
    m.color.r = 0.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.lifetime = rospy.Duration(0.5)
    return m

def publish_smoothed_track(track_id, track, result, pub, vis_pub):
    msg = PfGMFATrackArray()
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "vehicle"
    msg.header = header

    marker_array = MarkerArray()

    for i in range(len(track)):
        pose = result.atPose2(gtsam.symbol('x', i))
        t = track[i]

        m = PfGMFATrack()
        m.id = int(track_id)
        m.obj_class = t['type'] if 'type' in t else 1
        m.pos_x = pose.x()
        m.pos_y = pose.y()
        m.yaw = pose.theta()
        m.boundingbox = list(t['size'])[:3] + [0.0]*5
        m.confidence_ind = t.get('confidence', 0.5)
        msg.tracks.append(m)

        marker = create_marker(t, pose, 1000 + i, header)
        marker_array.markers.append(marker)

    pub.publish(msg)
    vis_pub.publish(marker_array)

def main():
    if len(sys.argv) < 2:
        print("Usage: smoother_main.py path_to_rosbag.bag")
        return

    bag_path = sys.argv[1]

    rospy.init_node("smoothing_result_publisher", anonymous=True)
    pub = rospy.Publisher("/smoothed_tracks", PfGMFATrackArray, queue_size=10)
    vis_pub = rospy.Publisher("/smoothed_track_markers", MarkerArray, queue_size=10)
    rate = rospy.Rate(10)

    try:
        tracks = load_tracks_from_bag(bag_path)

        for tid, tr in tracks.items():
            if rospy.is_shutdown():
                break
            if len(tr) < 3:
                continue
            print(f"Processing Track ID {tid} ({len(tr)} steps)")

            graph, initial = build_graph_from_track(tr)
            result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()

            publish_smoothed_track(tid, tr, result, pub, vis_pub)
            rate.sleep()

            for i in range(len(tr)):
                est = result.atPose2(gtsam.symbol('x', i))
                print(f" t={tr[i]['time']:.2f} → (x={est.x():.2f}, y={est.y():.2f}, yaw={est.theta():.2f})")

    except KeyboardInterrupt:
        print("\nSmoother interrupted by user. Shutting down cleanly.")

if __name__ == "__main__":
    main()
