#!/usr/bin/env python
import sys
import rospy
from actionlib_msgs.msg import GoalStatusArray
from std_srvs.srv import Trigger
from std_msgs.msg import Int8

class nav_client_node:
    def __init__(self):
        rospy.init_node('nav_client_node', anonymous=True)
        self.status_sub = rospy.Subscriber("/move_base/status", GoalStatusArray, self.callback_loop)
        self.loop_pub = rospy.Publisher("/loop_count", Int8, queue_size = 10)
        self.status = 0
        self.loop = 0

    def callback_loop(self, data):
        self.status = data.status_list[0]

        if self.status.status == 3:

            rospy.wait_for_service('start_wp_nav')
            try:
                service = rospy.ServiceProxy('start_wp_nav', Trigger)
                response = service()
            except rospy.ServiceException as e:
                print("Service call failed: %s" % e)

            self.loop += 1
        self.loop_pub.publish(self.loop)


if __name__ == "__main__":
    nc = nav_client_node()
    rospy.spin()
