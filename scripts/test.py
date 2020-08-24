#!/usr/bin/env python

import rospy
from actionlib_msgs.msg import GoalStatusArray
import numpy as np

class testNode:
    def __init__(self):
        
        rospy.init_node('testNode', anonymous=True)
        self.test_sub= rospy.Subscriber("/move_base/status", GoalStatusArray, self.callback_data)
        self.data =0

    def callback_data(self, data):
        self.data = data.status_list[0].status
        count = []
        next_count = []
        count.append(self.data)
        next_count.append(self.data)

#        split = (self.data[0],self.data[1])
        print(count, next_count)
#        rospy.loginfo("x:%f", GoalStatusArray)

if __name__ =='__main__':
    tn = testNode()
    rospy.spin()

