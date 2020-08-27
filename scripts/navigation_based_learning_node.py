#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('obstacle_avoidance')
import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from kobuki_msgs.msg import BumperEvent
from cv_bridge import CvBridge, CvBridgeError
from deep_learning import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Int8
from std_srvs.srv import Trigger
from actionlib_msgs.msg import GoalStatusArray
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import random
import math
import sys

class cource_following_learning_node:
	def __init__(self):
		rospy.init_node('cource_following_learning_node', anonymous=True)
		self.action_num = rospy.get_param("/LiDAR_based_learning_node/action_num", 3)
		print("action_num: " + str(self.action_num))
		self.dl = deep_learning(n_action = self.action_num)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
		self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
		self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
#		self.bumper_sub = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.callback_bumper)
		self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
		self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.callback_scan)
#		self.status_sub = rospy.Subscriber("/move_base/status", GoalStatusArray, self.callback_loop)
		self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
		self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
		self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
		self.action = 0
		self.reward = 0
		self.episode = 0
		self.count = 0
		self.status = 0
		self.loop_count = 0
		self.success = 0
		self.vel = Twist()
		self.cv_image = np.zeros((480,640,3), np.uint8)
		self.cv_left_image = np.zeros((480,640,3), np.uint8)
		self.cv_right_image = np.zeros((480,640,3), np.uint8)
		self.learning = True
		self.collision = False
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
			self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def callback_left_camera(self, data):
		try:
			self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def callback_right_camera(self, data):
		try:
			self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def callback_scan(self, scan):
		points = []
		angle = scan.angle_min
		for distance in scan.ranges:
			if distance != float('inf') and not math.isnan(distance):
				points.append((distance * math.cos(angle), distance * math.sin(angle)))
			angle += scan.angle_increment

			if distance <= 0.2:
				self.collision = True

	def callback_vel(self, data):
		self.vel = data
# action
		if self.vel.angular.z > 0.1:
			self.action = 1
		elif self.vel.angular.z < -0.1:
			self.action = 2
		else:
			self.action = 0

#	def callback_loop(self, data):
#		self.status = data.status_list[0]
#
#		if self.status.status == 3:
#			rospy.wait_for_service('start_wp_nav')
#			try:
#				service = rospy.ServiceProxy('start_wp_nav', Trigger)
#				response = service()
#			except rospy.ServiceException as e:
#				print("Service call failed: %s" % e)


	def callback_dl_training(self, data):
		resp = SetBoolResponse()
		self.learning = data.data
		resp.message = "Training: " + str(self.learning)
		resp.success = True
		return resp

	def loop(self):
		if self.cv_image.size != 640 * 480 * 3:
			return
		if self.cv_left_image.size != 640 * 480 * 3:
			return
		if self.cv_right_image.size != 640 * 480 * 3:
			return

		rospy.wait_for_service('/gazebo/get_model_state')
		get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		try:
			previous_model_state = get_model_state('mobile_base', 'world')
		except rospy.ServiceException as exc:
			print("Service did not process request: " + str(exc))
		if self.vel.linear.x == 0:
			return
		img = resize(self.cv_image, (48, 64), mode='constant')
		r, g, b = cv2.split(img)
		imgobj = np.asanyarray([r,g,b])

		img_left = resize(self.cv_left_image, (48, 64), mode='constant')
		r, g, b = cv2.split(img_left)
		imgobj_left = np.asanyarray([r,g,b])

		img_right = resize(self.cv_right_image, (48, 64), mode='constant')
		r, g, b = cv2.split(img_right)
		imgobj_right = np.asanyarray([r,g,b])

		ros_time = str(rospy.Time.now())

#		if self.episode >= 14:
#			self.learning = False

		if self.learning:
			if self.loop_count < 100:
				self.reward = 0
				action = self.dl.act_and_trains(imgobj, self.action)
				self.reward += 1 if action == self.action else 0
				if self.action == 0:
					action = self.dl.act_and_trains(imgobj_left, 2)
					self.reward += 1 if action == 2 else 0
					action = self.dl.act_and_trains(imgobj_right, 1)
					self.reward += 1 if action == 1 else 0
					self.success += self.reward / 3.0
				else:
					self.success += self.reward
				self.count += 1
				print(self.reward, end='')
				sys.stdout.flush()
				self.loop_count += 1
			else:
				action = self.dl.act_and_trains(imgobj, self.action)
				line = [str(self.episode), str(float(self.success) / self.count)]
				with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
					writer = csv.writer(f, lineterminator='\n')
					writer.writerow(line)
				print()
				print(" episode: " + str(self.episode) + ", success ratio: " + str(float(self.success)/self.count) + " " + str(self.count))
				self.count = 0
				self.success = 0
				self.loop_count = 0
				self.collision = False
				self.episode += 1
			self.action_pub.publish(self.action)

		else:
			self.action = self.dl.act(imgobj)
			print("TEST MODE: " + str(self.action))
			self.action_pub.publish(self.action)

		temp = copy.deepcopy(img)
		cv2.imshow("Resized Image", temp)
		temp = copy.deepcopy(img_left)
		cv2.imshow("Resized Left Image", temp)
		temp = copy.deepcopy(img_right)
		cv2.imshow("Resized Right Image", temp)
		cv2.waitKey(1)

if __name__ == '__main__':
	rg = cource_following_learning_node()
	DURATION = 0.2
	r = rospy.Rate(1 / DURATION)
	while not rospy.is_shutdown():
		rg.loop()
		r.sleep()
