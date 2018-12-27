#!/usr/bin/env python
# check node
# rostopic pub -1 /action std_msgs/Int8 1

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Int8

linear_vel = 0.5
angle_vel = 0.2

class robot_move:
	def __init__(self):
		rospy.init_node('robot_move', anonymous=True)
		self.action_sub = rospy.Subscriber("/action", Int8, self.callback_action)
		self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
		self.action = 0
		self.vel_msg = Twist()

	def callback_action(self, data):
		self.action = data.data

		# [velocity (m/s), angular velocity(rad/s)]
		velocity = [[linear_vel, 0.0], [linear_vel, angle_vel], [linear_vel, -angle_vel]]
		self.vel_msg.linear.x = velocity[self.action][0]
		self.vel_msg.angular.z = velocity[self.action][1]
		print 'velocity = ', self.vel_msg

		move = ['F', 'R', 'L']
		print 'action = ', move[self.action]

		#Publish the velocity
		self.vel_pub.publish(self.vel_msg)

if __name__ == '__main__':
	rm = robot_move()
	
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting Down")
