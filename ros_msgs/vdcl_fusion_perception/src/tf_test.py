import rospy
# from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

class TF_reader:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('tf_reader', anonymous=True)
        
        # Set up a subscriber to listen to the TF messages
        self.tf_subscriber = rospy.Subscriber('/tf', TFMessage, self.callback)

    def callback(self, data):
        # Process the TF message data
        for transform in data.transforms:
            rospy.loginfo("Received transform: %s", transform)

if __name__ == '__main__':
    try:
        tf_reader = TF_reader()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass