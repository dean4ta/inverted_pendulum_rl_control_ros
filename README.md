# Inverted Pendulum RL Control Packages

* Author: Dean Fortier <dean4ta@gmail.com>
* License: GNU General Public License, version 3 (GPL-3.0)

This repository contains the RL Control package for an [Inverted Pendulum Simulation](https://github.com/dean4ta/gazebo_ros_demos).

## Quick Start

Start the partnered simulation:

    roslaunch rrbot_gazebo rrbot_world.launch rviz:=true

Start the rl training node:
    
    rosrun inverted_pendulum_rl_control train_ddpg.py

