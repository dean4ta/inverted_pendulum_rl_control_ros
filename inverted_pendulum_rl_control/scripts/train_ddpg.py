import rospy
import rospkg
from rl_common.ddpg import ActorNet, CriticNet, Memory, Agent

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from inverted_pendulum_rl_control.srv import (
    SaveModel,
    SaveModelResponse,
    SaveModelRequest,
)

from gazebo_msgs.srv import (
    SetModelConfiguration,
    SetModelConfigurationRequest,
)

import numpy as np
import os
import torch
from collections import namedtuple

TrainingRecord = namedtuple("TrainingRecord", ["ep", "reward"])
Transition = namedtuple("Transition", ["s", "a", "r", "s_"])


class TrainDDPG:
    def __init__(self):
        # TODO: parameterize the namespace
        rospy.loginfo("Initializing DDPG")
        self.joint_sub = rospy.Subscriber(
            "/rrbot/joint_states", JointState, self.joint_callback
        )
        self.action_pub = rospy.Publisher(
            "/rrbot/joint1_effort_controller/command", Float64, queue_size=10
        )
        self.reward_pub = rospy.Publisher("/reward", Float64, queue_size=10)
        self.state_pub = rospy.Publisher("/state", Float64MultiArray, queue_size=10)
        self.save_model_srv = rospy.Service("save_model", SaveModel, self.save_model)

        rospy.wait_for_service("/gazebo/set_model_configuration")
        self.reset()
        self.agent = Agent()
        self.is_state_ready = False
        self.state = np.zeros(3)
        rospy.loginfo("Beginning Training")
        self.train()

    def joint_callback(self, msg):
        position = self.angle_normalize(msg.position[0])
        velocity = msg.velocity[0]
        effort = msg.effort[0]
        self.state = np.array([position, velocity, effort])
        self.is_state_ready = True

    def angle_normalize(self, angle):
        # Normalize angle to -pi to pi
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def publish_state(self, state):
        msg = Float64MultiArray()
        msg.data = state
        self.state_pub.publish(msg)

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

    def get_reward(self, state_next):
        reward = 0
        angle = self.angle_normalize(state_next[0])
        reward += 40 * np.exp(-np.abs(angle))

        if np.abs(state_next[2]) > 10:
            reward -= 20 * np.abs(state_next[2])

        self.reward_pub.publish(Float64(reward))
        return reward

    def save_model(self, req):
        rospack = rospkg.RosPack()
        try:
            file_path = rospack.get_path("inverted_pendulum_rl_control") + "/models/"
            file_str = file_path + req.filename
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            torch.save(
                self.agent.eval_anet.state_dict(),
                file_str + "_actor.pkl",
            )
            torch.save(
                self.agent.eval_cnet.state_dict(),
                file_str + "_critic.pkl",
            )
            rospy.loginfo("Saved model as models/" + req.filename)
            return SaveModelResponse(0)
        except Exception as e:
            rospy.logwarn("Failed to save model: %s" % e)
            return SaveModelResponse(-1)

    # TODO: Restructure function to be callback based so process can be successfully exited with Ctrl+C
    def train(self):
        training_records = []
        running_reward, running_q = -1000, 0
        self.episodes = 0
        for episode in range(750):
            self.episodes = episode
            if rospy.is_shutdown():
                return
            score = 0
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
                action = self.agent.select_action(state)[0]
                action_cap = np.clip(action, -50, 50)

                self.action_pub.publish(Float64(action_cap))
                self.publish_state(state)

                # get next state
                for _ in range(2):  # allow time to pass before next state
                    while self.is_state_ready == False:
                        pass
                    self.is_state_ready = False
                state_next = self.state
                reward = self.get_reward(state_next)
                if t < 5:  # first few rewards seem to be random
                    reward = 0
                score += reward

                self.agent.store_transition(
                    Transition(state, action, reward, state_next)
                )
                state = state_next
                if self.agent.memory.isfull:
                    q = self.agent.update()
                    running_q = 0.99 * running_q + 0.01 * q
                t += 1

            running_reward = running_reward * 0.9 + score * 0.1
            training_records.append(TrainingRecord(episode, running_reward))

            rospy.loginfo(
                "Episode: %d, score: %f, running_reward: %f, running_q: %f"
                % (episode, score, running_reward, running_q)
            )


if __name__ == "__main__":
    rospy.init_node("train_ddpg")
    TrainDDPG()
    rospy.spin()
