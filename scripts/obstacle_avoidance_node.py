#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('obstacle_avoidance')
import rospy
import cv2
from sensor_msgs.msg import Image
from kobuki_msgs.msg import BumperEvent
from cv_bridge import CvBridge, CvBridgeError
from reinforcement_learning import *
from skimage.transform import resize
from std_msgs.msg import Float32, Int8
from std_srvs.srv import Empty
import sys
import skimage.transform
import csv
import os
import time
import copy

class obstacle_avoidance_node:
	def __init__(self):
		rospy.init_node('obstacle_avoidance_node', anonymous=True)
		self.action_num = rospy.get_param("/obstacle_avoidance_node/action_num", 3)
		print("action_num: " + str(self.action_num))
		self.rl = reinforcement_learning(n_action = self.action_num)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.callback)
#		self.bumper_sub = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.callback_bumper)
#		self.reward_sub = rospy.Subscriber("/reward", Float32, self.callback_reward)
		self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
		self.action = 0
		self.reward = 0
		self.cv_image = np.zeros((480,640,3), np.uint8)
		self.count = 0
		self.learning = True
		self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
		self.action_list = ['Front', 'Right', 'Left']
		self.path = '~/result'
		os.makedirs(self.path + self.start_time)

		with open(self.path + self.start_time + '/' +  'reward.csv', 'w') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(['rostime', 'reward', 'action'])
		self.done = False

	def callback(self, data):
		try:
			img = self.bridge.imgmsg_to_cv2(data, 'passthrough')
			depth_array = np.array(img, dtype=np.float32) * 50 #check magic number
			self.cv_image = depth_array.astype(np.uint8)
		except CvBridgeError as e:
			print(e)

#		temp = copy.deepcopy(self.cv_image)
#		cv2.imshow("Capture Image", temp)
#		cv2.waitKey(1)


#	def callback_bumper(self, bumper)
#		self.action = self.rl.stop_episode_and_train(imgobj, self.reward, self.done)
#		rospy.wait_for_service('/gazebo/reset_world')
#		reset_world = rospy.ServiceProxy('/gazebo/reset_world',Empty)

	def loop(self):
		print(self.cv_image.size)
		if self.cv_image.size != 640 * 480:
			return

		self.reward = 1
		img = resize(self.cv_image, (48, 64), mode='constant')
		print(img)
		imgobj = np.asanyarray([img])

		self.learning = True

		ros_time = str(rospy.Time.now())
		if self.learning:
			self.count += 1
			if self.count % 30 == 0:
				self.done = True
			if self.done:
				self.action = self.rl.stop_episode_and_train(imgobj, self.reward, self.done)
				self.done = False
				print('Last step in this episode')
			else:
				self.action = self.rl.act_and_trains(imgobj, self.reward)

			line = [ros_time, str(self.reward), str(self.action)]
			with open(self.path + self.start_time + '/' +  'reward.csv', 'a') as f:
				writer = csv.writer(f, lineterminator='\n')
				writer.writerow(line)

		else:
			self.action = self.rl.act(imgobj)
		self.action_pub.publish(self.action)

		print("learning = " + str(self.learning) + " count: " + str(self.count) + " action: " + str(self.action) + ", reward: " + str(round(self.reward,5)))
		temp = copy.deepcopy(img)
		cv2.imshow("Resized Image", temp)
		cv2.waitKey(1)

if __name__ == '__main__':
	rg = obstacle_avoidance_node()
	DURATION = 1.0
	r = rospy.Rate(1 / DURATION)
	while not rospy.is_shutdown():
		rg.loop()
		r.sleep()
