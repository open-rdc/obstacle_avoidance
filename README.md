# obstacle_avoidance

### インストール

```
sudo apt-get install ros-kinetic-turtlebot-gazebo
sudo apt-get install ros-kinetic-turtlebot-rviz-launchers
```

### 環境設定

```
cp my_stage01.world /opt/ros/kinetic/share/turtlebot_gazebo/worlds
```

### 実行

```
export TURTLEBOT_GAZEBO_WORLD_FILE=/opt/ros/kinetic/share/turtlebot_gazebo/worlds/my_stage01.world

roslaunch turtlebot_gazebo turtlebot_world.launch

roslaunch turtlebot_rviz_launchers view_robot.launch
```
