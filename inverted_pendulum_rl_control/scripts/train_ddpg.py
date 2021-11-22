import rospy
import rospkg
from rl_common.ddpg import Agent

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from gazebo_msgs.msg import ModelStates
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from std_srvs.srv import Empty, EmptyRequest
from inverted_pendulum_rl_control.srv import (
    SaveModel,
    SaveModelResponse,
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
N_JOINTS = 12
S_PER_JOINT = 3
MAX_RAND_Torque = 20


class TrainDDPG:
    def __init__(self):
        # TODO: parameterize the namespace
        rospy.loginfo("Initializing DDPG")

        self.init_parameters()
        self.init_subscribers()
        self.init_publishers()

        self.save_model_srv = rospy.Service("save_model", SaveModel, self.save_model)

        self.wait_for_services()
        self.reset()
        self.agent = Agent(n_states=N_JOINTS * S_PER_JOINT, n_actions=N_JOINTS)
        self.is_state_ready = False
        self.state = np.zeros(N_JOINTS * S_PER_JOINT)
        rospy.loginfo("Beginning Training")
        self.train()

    def init_parameters(self):
        self.joint_names = rospy.get_param("/joint_names")
        self.standing_joint_position = rospy.get_param("standing_joint_position")
        self.standing_joint_effort = rospy.get_param("standing_joint_effort")
        self.start_episode_standing = rospy.get_param("start_episode_standing")
        self.use_effort = rospy.get_param("use_effort")
        self.expert_system_command_topic = rospy.get_param(
            "champ_controller/joint_controller_topic"
        )

    def init_subscribers(self):
        self.joint_sub = rospy.Subscriber(
            "/joint_states", JointState, self.joint_callback
        )
        self.model_state_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_state_callback
        )
        self.expert_system_command_sub = rospy.Subscriber(
            self.expert_system_command_topic,
            JointTrajectory,
            self.expert_system_callback,
        )

    def init_publishers(self):
        self.init_joint_publishers()
        self.reward_pub = rospy.Publisher("/reward", Float64, queue_size=10)
        self.state_pub = rospy.Publisher("/state", Float64MultiArray, queue_size=10)

    def init_joint_publishers(self):
        if self.use_effort:
            # fmt: off
            self.LF_HAA_pub = rospy.Publisher("/joint_LF_HAA_effort_controller/command", Float64, queue_size=10)
            self.LF_HFE_pub = rospy.Publisher("/joint_LF_HFE_effort_controller/command", Float64, queue_size=10)
            self.LF_KFE_pub = rospy.Publisher("/joint_LF_KFE_effort_controller/command", Float64, queue_size=10)
            self.LH_HAA_pub = rospy.Publisher("/joint_LH_HAA_effort_controller/command", Float64, queue_size=10)
            self.LH_HFE_pub = rospy.Publisher("/joint_LH_HFE_effort_controller/command", Float64, queue_size=10)
            self.LH_KFE_pub = rospy.Publisher("/joint_LH_KFE_effort_controller/command", Float64, queue_size=10)
            self.RF_HAA_pub = rospy.Publisher("/joint_RF_HAA_effort_controller/command", Float64, queue_size=10)
            self.RF_HFE_pub = rospy.Publisher("/joint_RF_HFE_effort_controller/command", Float64, queue_size=10)
            self.RF_KFE_pub = rospy.Publisher("/joint_RF_KFE_effort_controller/command", Float64, queue_size=10)
            self.RH_HAA_pub = rospy.Publisher("/joint_RH_HAA_effort_controller/command", Float64, queue_size=10)
            self.RH_HFE_pub = rospy.Publisher("/joint_RH_HFE_effort_controller/command", Float64, queue_size=10)
            self.RH_KFE_pub = rospy.Publisher("/joint_RH_KFE_effort_controller/command", Float64, queue_size=10)
            # fmt: on
        else:
            self.joint_position_pub = rospy.Publisher(
                "/joint_group_position_controller/command",
                JointTrajectory,
                queue_size=10,
            )
            self.joint_seq = 0

    def wait_for_services(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/reset_world")
        rospy.wait_for_service("/gazebo/set_model_configuration")
        rospy.wait_for_service("/gazebo/unpause_physics")

    def joint_callback(self, msg):
        position = np.array(msg.position)
        velocity = np.array(msg.velocity)
        effort = np.array(msg.effort)
        self.state = np.concatenate((position, velocity, effort))
        self.is_state_ready = True

    def model_state_callback(self, msg):
        self.model_state_pose = msg.pose[1]
        self.model_state_twist = msg.twist[1]

    def expert_system_callback(self, msg):
        self.expert_system_command = msg.points[0].positions

    def angle_normalize(self, angle):
        # Normalize angle to -pi to pi
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def publish_state(self, state):
        msg = Float64MultiArray()
        msg.data = state
        self.state_pub.publish(msg)

    def publish_action(self, action):
        if self.use_effort:
            self.LF_HAA_pub.publish(Float64(action[0]))
            self.LF_HFE_pub.publish(Float64(action[1]))
            self.LF_KFE_pub.publish(Float64(action[2]))
            self.LH_HAA_pub.publish(Float64(action[3]))
            self.LH_HFE_pub.publish(Float64(action[4]))
            self.LH_KFE_pub.publish(Float64(action[5]))
            self.RF_HAA_pub.publish(Float64(action[6]))
            self.RF_HFE_pub.publish(Float64(action[7]))
            self.RF_KFE_pub.publish(Float64(action[8]))
            self.RH_HAA_pub.publish(Float64(action[9]))
            self.RH_HFE_pub.publish(Float64(action[10]))
            self.RH_KFE_pub.publish(Float64(action[11]))
        else:
            joint_trajectory = JointTrajectory()
            joint_trajectory.header.seq = self.joint_seq
            self.joint_seq += 1
            joint_trajectory.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            point.positions = action
            joint_trajectory.points.append(point)
            joint_trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.01)
            self.joint_position_pub.publish(joint_trajectory)

    def reset(self):
        # Reset performs the following actions:
        # 1. Pause physics
        # 2. Reset the world
        # 3. Set the model state
        # 4. Unpause physics

        # 1. Pause physics
        req_pause = EmptyRequest()
        try:
            rospy.ServiceProxy("/gazebo/pause_physics", Empty)(req_pause)
        except rospy.ServiceException as e:
            rospy.logwarn("/gazebo/pause_physics service call failed: {0}".format(e))

        # 2. Reset the world
        req_reset_world = EmptyRequest()
        try:
            rospy.ServiceProxy("/gazebo/reset_world", Empty)(req_reset_world)
        except rospy.ServiceException as e:
            rospy.logwarn("/gazebo/reset_world service call failed: {0}".format(e))

        # 3. Set the model state
        req = SetModelConfigurationRequest()
        req.model_name = "/"
        req.urdf_param_name = "robot_description"
        req.joint_names = self.joint_names
        if not self.start_episode_standing:
            joint_positions = np.random.uniform(-9.42477796077, 9.42477796077, (4, 3))
            joint_positions[:, 0] = np.random.uniform(-0.72, 0.49, (4,))
            joint_positions = joint_positions.flatten()
        else:
            joint_positions = self.standing_joint_position
            # joint_positions += np.random.normal(0, 0.01, (N_JOINTS,))

        req.joint_positions = joint_positions
        try:
            resp = rospy.ServiceProxy(
                "/gazebo/set_model_configuration", SetModelConfiguration
            )(req)
            if not resp.success:
                rospy.logwarn("Failed to reset the robot")
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)

        # 4. Unpause physics
        req_unpause = EmptyRequest()
        try:
            rospy.ServiceProxy("/gazebo/unpause_physics", Empty)(req_unpause)
        except rospy.ServiceException as e:
            rospy.logwarn("/gazebo/unpause_physics service call failed: {0}".format(e))

        if not self.start_episode_standing:
            joint_efforts = np.random.uniform(
                -MAX_RAND_Torque, MAX_RAND_Torque, (N_JOINTS,)
            )
        else:
            joint_efforts = self.standing_joint_effort
            joint_efforts += np.random.normal(0, 0.5, (N_JOINTS,))

        self.publish_action(joint_efforts)

    def get_reward(self, state_next):
        if not self.start_episode_standing:
            reward = 0
            for joint in range(N_JOINTS):
                reward += np.exp(
                    -np.abs(state_next[joint] - self.standing_joint_position[joint])
                )
            self.reward_pub.publish(Float64(reward))
            return reward
        else:
            reward = self.model_state_pose.position.z * 10
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
        self.episode = 0
        tot_episodes = 10000
        rospy.loginfo("Executing %s episodes" % tot_episodes)
        for episode in range(tot_episodes):
            self.episode = episode
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

                if t > 20:
                    action = self.agent.select_action(state)[0]
                else:
                    action = self.expert_system_command

                self.publish_action(action)
                self.publish_state(state)

                # get next state
                for _ in range(2):  # allow time for state to react to action
                    while self.is_state_ready == False:
                        pass
                    self.is_state_ready = False
                state_next = self.state
                reward = self.get_reward(state_next)
                if t < 2:  # first few rewards seem to be random so we ignore them
                    reward = 0
                score += reward

                self.agent.store_transition(
                    Transition(state, action, reward, state_next)
                )
                state = state_next
                if self.agent.memory.isfull:
                    rospy.loginfo("Memory is full, updating agent!")
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
