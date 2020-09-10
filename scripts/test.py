#!/usr/bin/env python

import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32
from nav_msgs.msg import Path
import math
import numpy as np

class testNode:
    def __init__(self):
        rospy.init_node('testNode', anonymous=True)
        self.amcl_pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_amcl_pose)
        self.nav_path_sub = rospy.Subscriber("/move_base/DWAPlannerROS/local_plan", Path, self.callback_nav_path)
        self.distance_pub = rospy.Publisher("distance", Float32, queue_size=1)
        self.data = 0
        self.distance = 0
        self.amcl_pose = 0
        self.nav_path = 0

    def callback_amcl_pose(self, data):
        self.amcl_pose = data

    def callback_nav_path(self, data):
        self.nav_path = data

    def callback_modify_action(self):

        diff_x = self.amcl_pose.pose.pose.position.x - self.nav_path.poses[0].pose.position.x
        diff_y = self.amcl_pose.pose.pose.position.y - self.nav_path.poses[0].pose.position.y
        self.distance = math.sqrt(diff_x ** 2 + diff_y ** 2)
        print(self.distance)
        self.distance_pub.publish(self.distance)

#    def callback_data(self, data):
#        self.data = data.pose.pose.position.x
#        self.data = data.poses[0].pose.position.x
#        print(self.data)
#        rospy.loginfo("x:%f", GoalStatusArray)

if __name__ =='__main__':
    tn = testNode()
    rospy.spin()
