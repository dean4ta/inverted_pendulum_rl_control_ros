# Inverted Pendulum RL Control Packages

* Author: Dean Fortier <dean4ta@gmail.com>
* License: GNU General Public License, version 3 (GPL-3.0)

This repository contains the RL Control package for the [Inverted Pendulum Simulation](https://github.com/dean4ta/gazebo_ros_demos).

## Quick Start

Start the partnered simulation:

    roslaunch rrbot_gazebo rrbot_world.launch rviz:=true

Start the rl training node:
    
    rosrun inverted_pendulum_rl_control train_ddpg.py

## Saving and Loading Model

Once training has produced a model, you can save it to a file with with the following command:

    rosservice call /save_model "filename: 'model_name'"

This command saves the model to the `inverted_pendulum_rl_control/models/` folder.

You can load and evaluate a model with the following command:

    roslaunch inverted_pendulum_rl_control eval.launch

This repo contains an existing model that can be evaluated like so:

    roslaunch inverted_pendulum_rl_control eval.launch rl_model_name:=trained_actor.pkl

## Reinforcement Learning Source

The DDPG algorithm used was inspired by the [simple-pytorch-rl](https://github.com/xtma/simple-pytorch-rl.git) repo.
