import rospy
import rospkg
from rl_common.ddpg import InferenceAgent

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from gazebo_msgs.srv import (
    SetModelConfiguration,
    SetModelConfigurationRequest,
)

import numpy as np
import os
import torch


class EvalDDPG:
    def __init__(self):
        # TODO: parameterize the namespace
        rospy.loginfo("Initializing DDPG Evaluation")
        self.joint_sub = rospy.Subscriber(
            "/rrbot/joint_states", JointState, self.joint_callback
        )
        self.action_pub = rospy.Publisher(
            "/rrbot/joint1_effort_controller/command", Float64, queue_size=10
        )

        rospy.wait_for_service("/gazebo/set_model_configuration")
        self.reset()
        self.controller = InferenceAgent()
        self.is_state_ready = False
        self.state = np.zeros(3)
        rospy.loginfo("Beginning Evaluation")
        self.eval()

    def joint_callback(self, msg):
        position = self.angle_normalize(msg.position[0])
        velocity = msg.velocity[0]
        effort = msg.effort[0]
        self.state = np.array([position, velocity, effort])
        self.is_state_ready = True

    def angle_normalize(self, angle):
        # Normalize angle to -pi to pi
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        req = SetModelConfigurationRequest()
        req.model_name = "rrbot"
        req.urdf_param_name = "robot_description"
        req.joint_names = ["joint1"]
        req.joint_positions = [np.random.uniform(-np.pi, np.pi)]
        try:
            resp = rospy.ServiceProxy(
                "/gazebo/set_model_configuration", SetModelConfiguration
            )(req)
            if not resp.success:
                rospy.logwarn("Failed to reset the robot")
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
        self.action_pub.publish(Float64(np.random.uniform(-20, 20)))

    # TODO: Restructure function to be callback based so process can be successfully exited with Ctrl+C
    def eval(self):
        self.episodes = 0
        for episode in range(100):
            self.episodes = episode
            if rospy.is_shutdown():
                return
            self.reset()
            # get first state for the episode
            while self.is_state_ready == False:
                pass
            self.is_state_ready = False
            state = self.state
            # action_prev = 0

            t = 0
            while t < 200:
                if rospy.is_shutdown():
                    return
                action = self.controller.select_action(state)[0]

                self.action_pub.publish(Float64(action))

                # get next state
                while self.is_state_ready == False:
                    pass
                self.is_state_ready = False
                state_next = self.state

                state = state_next
                t += 1

            rospy.loginfo("Episode: %d" % (episode))


if __name__ == "__main__":
    rospy.init_node("train_ddpg")
    EvalDDPG()
    rospy.spin()
