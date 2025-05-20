import rospy
import numpy as np

#lidar subscriptionnode
from sensor_msgs.msg import PointCloud2

class LidarSubscriber:
    def __init__(self):
        # Initialize the node
        rospy.init_node('lidar_subscriber', anonymous=True)

        # Subscribe to the PointCloud2 topic
        self.subscriber = rospy.Subscriber('/velodyne_FC/velodyne_points', PointCloud2, self.callback)

    def callback(self, msg):
        # Process the PointCloud2 data
        rospy.loginfo("Received Lidar data")
        point_dtype = np.dtype([
            ('timestampSec', np.uint32),
            ('timestampNsec', np.uint32),
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.uint8),
            ('ring', np.uint8),
            ('_padding', 'V2'),  # 22 bytes align 맞추기 위해 2바이트 패딩 (optional)
        ])      
        data_points = np.frombuffer(msg.data, dtype=point_dtype)
        x_data = data_points['x']
        y_data = data_points['y']
        z_data = data_points['z']
        intensity_data = data_points['intensity']
        point_array = np.array([x_data, y_data, z_data, intensity_data]).T
        print(point_array.shape)
        print(str(data_points['timestampSec'][0])+"."+str(data_points['timestampNsec'][0]))
        print(msg.header.stamp)

if __name__ == '__main__':
    lidar_subscriber = LidarSubscriber()
    rospy.spin()