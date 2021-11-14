import rospy
from rl_common.ddpg import ActorNet, CriticNet, Memory, Agent

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from gazebo_msgs.srv import (
    SetModelConfiguration,
    SetModelConfigurationRequest,
    SetModelConfigurationResponse,
)

import numpy as np
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
        rospy.wait_for_service("/gazebo/set_model_configuration")
        self.reset()
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
        # TODO: consider adding a random torque to the robot as well

    def get_reward(self, state_next):
        reward = 0
        angle = self.angle_normalize(state_next[0])
        reward += 40 * np.exp(-np.abs(angle))
        if np.abs(angle) < np.pi / 2 and self.episodes > 50:
            reward += 5 * np.exp(-np.abs(state_next[1]))
            reward += 5 * np.exp(-np.abs(state_next[2]))

        if np.abs(state_next[2]) > 10:
            reward -= 20 * np.abs(state_next[2])

        self.reward_pub.publish(Float64(reward))
        return reward

    # TODO: Restructure function to be callback based so process can be successfully exited with Ctrl+C
    def train(self):
        agent = Agent()

        training_records = []
        running_reward, running_q = -1000, 0
        self.episodes = 0
        for episode in range(500):
            self.episodes = episode
            if rospy.is_shutdown():
                return
            max_action, min_action = 0, 0
            score = 0
            self.reset()
            # get first state for the episode
            while self.is_state_ready == False:
                pass
            self.is_state_ready = False
            state = self.state

            for t in range(500):
                if rospy.is_shutdown():
                    return
                action = agent.select_action(state)[0]
                # if np.abs(action) > :
                #     continue  # terminate the episode
                self.action_pub.publish(Float64(action))

                # get next state
                while self.is_state_ready == False:
                    pass
                self.is_state_ready = False
                state_next = self.state
                reward = self.get_reward(state_next)
                score += reward

                if max_action < action:
                    max_action = action
                if min_action > action:
                    min_action = action

                agent.store_transition(Transition(state, action, reward, state_next))
                state = state_next
                if agent.memory.isfull:
                    q = agent.update()
                    running_q = 0.99 * running_q + 0.01 * q

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
