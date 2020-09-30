#!/usr/bin/env python

import rospy
from actionlib_msgs.msg import GoalStatusArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class testNode:
    def __init__(self):
        rospy.init_node('testNode', anonymous=True)
        self.pose_sub= rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub= rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.pose = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.path = 0.0
        self.distance = 0.0
        self.distance_list = []

    def callback_path(self, data):
        self.path = data

    def callback_pose(self, data):
        self.pose = data.pose.pose
        self.pose_x = self.pose.position.x
        self.pose_y = self.pose.position.y

        for i in range(len(self.path.poses)):
            self.path_x = self.path.poses[i].pose.position.x
            self.path_y = self.path.poses[i].pose.position.y
            self.distance = np.sqrt((self.pose_x - self.path_x)**2 - (self.pose_y - self.path_y)**2)
            self.distance_list.append(self.distance)

        print(min(self.distance_list))


if __name__ =='__main__':
    tn = testNode()
    rospy.spin()

