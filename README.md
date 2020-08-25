# obstacle_avoidance

### Build

```
rosdep install obstacle_avoidance
catkin build obstacle_avoidance
```

### Execution

```
roslaunch obstacle_avoidance LiDAR_based_learning_sim.launch
```

### Checking Results

```
cd ~/.ros/data
```

![Screenshot 2020-04-12 11:54:06](https://user-images.githubusercontent.com/5755200/79059403-87a64600-7cb4-11ea-894c-1d5d825748a6.png)

### Navigation based
bring up gazebo and learning
```
roslaunch obstacle_avoidance navigation_based_learning_sim.launch
```

 /turtlebot3_navigation/launch/move_base.launchを変更
```
<arg name="cmd_vel_topic" default="/cmd_vel" /> --> <arg name="cmd_vel_topic" default="/nav_vel" />　
 
```

bring up navigation
```
roslaunch obstacle_avoidance turtlebot3_navigation.launch map_file:={/YOUR_PATH/maps/willowgarage.yaml} waypoints_file:={/YOUR_PATH/maps/willow_loop.yaml}
```

start waypoint navigation
```
rosservice call /start_wp_nav
```
