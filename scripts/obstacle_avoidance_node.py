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
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
import csv
import os
import time
import copy
import random

class obstacle_avoidance_node:
	def __init__(self):
		rospy.init_node('obstacle_avoidance_node', anonymous=True)
		self.action_num = rospy.get_param("/obstacle_avoidance_node/action_num", 3)
		print("action_num: " + str(self.action_num))
		self.rl = reinforcement_learning(n_action = self.action_num)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.callback)
		self.bumper_sub = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.callback_bumper)
		self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
		self.action = 0
		self.reward = 0
		self.episode = 0
		self.cv_image = np.zeros((480,640,3), np.uint8)
		self.learning = True
		self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
		self.action_list = ['Front', 'Right', 'Left']
		self.path = 'data/result'
		self.previous_reset_time = 0
		self.start_time_s = rospy.get_time()
		os.makedirs(self.path + self.start_time)

		with open(self.path + self.start_time + '/' +  'reward.csv', 'w') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(['epsode', 'time(s)', 'reward'])

	def callback(self, data):
		try:
			img = self.bridge.imgmsg_to_cv2(data, 'passthrough')
			depth_array = np.array(img, dtype=np.float32) * 50 #check magic number
			self.cv_image = depth_array.astype(np.uint8)
		except CvBridgeError as e:
			print(e)

	def reset_simulation(self):
		rospy.wait_for_service('/gazebo/set_model_state')
		set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
		model_state = ModelState()
		model_state.model_name = 'mobile_base'
		model_state.pose.position.x = 0
		model_state.pose.position.y = 0
		model_state.pose.position.z = 0
		model_state.pose.orientation.x = 0
		model_state.pose.orientation.y = 0
		model_state.pose.orientation.z = random.random() - 0.5
		model_state.pose.orientation.w = 1
		model_state.twist.linear.x = 0
		model_state.twist.linear.y = 0
		model_state.twist.linear.z = 0
		model_state.twist.angular.x = 0
		model_state.twist.angular.y = 0
		model_state.twist.angular.z = 0
		model_state.reference_frame = 'world'
		try:
			set_model_state(model_state)
		except rospy.ServiceException as exc:
			print("Service did not process request: " + str(exc))


	def callback_bumper(self, bumper):
		if rospy.get_time() - self.previous_reset_time < 1:
			return
		self.previous_reset_time = rospy.get_time()
		print("!!!!!!! RESET !!!!!!!")

		self.reset_simulation()
		reward = -1
		img = resize(self.cv_image, (48, 64), mode='constant')
		imgobj = np.asanyarray([img])
		self.action = self.rl.stop_episode_and_train(imgobj, reward, False)
		print("learning = " + str(self.learning) + " episode: " + str(self.episode) + " action: " + str(self.action) + ", reward: " + str(round(reward,5)))

		line = [str(self.episode), str(rospy.get_time() - self.start_time_s), str(reward)]
		with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(line)
		self.episode += 1

	def loop(self):
		if self.cv_image.size != 640 * 480:
			return

		rospy.wait_for_service('/gazebo/get_model_state')
		get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		try:
			previous_model_state = get_model_state('mobile_base', 'world')
		except rospy.ServiceException as exc:
			print("Service did not process request: " + str(exc))
		img = resize(self.cv_image, (48, 64), mode='constant')
		imgobj = np.asanyarray([img])

		self.learning = True

		ros_time = str(rospy.Time.now())

		if self.learning:
			if previous_model_state.pose.position.x < 25:
				reward = 0
				self.action = self.rl.act_and_trains(imgobj, reward)
			else:
				reward = 1
				self.action = self.rl.stop_episode_and_train(imgobj, reward, False)
				line = [str(self.episode), str(rospy.get_time() - self.start_time_s), str(reward)]
				with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
					writer = csv.writer(f, lineterminator='\n')
					writer.writerow(line)
				self.reset_simulation()
				self.episode += 1

		else:
			self.action = self.rl.act(imgobj)
		self.action_pub.publish(self.action)

		print("learning = " + str(self.learning) + " episode: " + str(self.episode) + " action: " + str(self.action) + ", reward: " + str(round(reward,5)))
		temp = copy.deepcopy(img)
		cv2.imshow("Resized Image", temp)
		cv2.waitKey(1)

if __name__ == '__main__':
	rg = obstacle_avoidance_node()
	DURATION = 0.1
	r = rospy.Rate(1 / DURATION)
	while not rospy.is_shutdown():
		rg.loop()
		r.sleep()
