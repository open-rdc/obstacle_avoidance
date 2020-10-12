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
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
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
		self.action_num = rospy.get_param("/LiDAR_based_learning_node/action_num", 1)
		print("action_num: " + str(self.action_num))
		self.dl = deep_learning(n_action = self.action_num)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
		self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
		self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
#		self.bumper_sub = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.callback_bumper)
		self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
		self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.callback_scan)
		self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
		self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
		self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
                self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
		self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
		self.pose = 0.0
		self.pose_x = 0.0
		self.pose_y = 0.0
		self.path_pose = 0.0
		self.distance = 0.0
		self.min_distance = 0.0
		self.action = 0.0
		self.reward = 0
		self.episode = 0
		self.count = 0
		self.status = 0
		self.loop_count = 0
		self.success = 0.0
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
		self.select_dl_out = False
		self.correct_count = 0
		self.incorrect_count = 0
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

        def callback_path(self, data):
        	self.path_pose = data
	def callback_pose(self, data):
		self.distance_list = []
		self.pose = data.pose.pose
		self.pose_x = self.pose.position.x
		self.pose_y = self.pose.position.y

		for i in range(len(self.path_pose.poses)):
			self.path_x = self.path_pose.poses[i].pose.position.x
			self.path_y = self.path_pose.poses[i].pose.position.y
			self.distance = np.sqrt(abs((self.pose_x - self.path_x)**2 + (self.pose_y - self.path_y)**2))
			self.distance_list.append(self.distance)
			self.min_distance = min(self.distance_list)

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
		self.action = self.vel.angular.z

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

                if self.learning:
		        action = self.dl.act_and_trains(imgobj, self.action)
			if abs(self.action) < 0.1:
			        action_left = self.dl.act_and_trains(imgobj_left, self.action - 0.2)
				action_right = self.dl.act_and_trains(imgobj_right, self.action + 0.2)

			print(" episode: " + str(self.episode) + ", success ratio: " + str(self.success))
			line = [str(self.episode), str(self.success)]
			with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
				writer = csv.writer(f, lineterminator='\n')
				writer.writerow(line)

			self.success += abs(action - self.action)
			self.episode += 1
			print(" episode: " + str(self.episode) + ", success ratio: " + str(self.success))

			self.vel.linear.x = 0.2
			self.vel.angular.z = self.action
			self.nav_pub.publish(self.vel)

		else:
			self.action = self.dl.act(imgobj)
			print("TEST MODE: " + str(self.action))
			self.vel.linear.x = 0.2
			self.vel.angular.z = self.action
			self.nav_pub.publish(self.vel)

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
