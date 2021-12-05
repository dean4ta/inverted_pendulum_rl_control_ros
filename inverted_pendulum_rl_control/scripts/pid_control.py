import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from gazebo_msgs.srv import (
    SetModelConfiguration,
    SetModelConfigurationRequest,
)

import numpy as np


class PIDControl:
    def __init__(self):
        rospy.loginfo("Initializing PID Control")
        self.joint_sub = rospy.Subscriber(
            "/rrbot/joint_states", JointState, self.joint_callback
        )
        self.action_pub = rospy.Publisher(
            "/rrbot/joint1_effort_controller/command", Float64, queue_size=10
        )
        self.PID = PIDController(kp=50.0, ki=0.01, kd=20.0)
        self.desired_position = 0.0
        self.iteration = 0

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
        self.desired_position = np.random.uniform(-np.pi, np.pi)

    def joint_callback(self, msg):
        output = self.PID.update(self.desired_position - msg.position[0], 1 / 50.0)
        output = np.clip(output, -50, 50)
        self.action_pub.publish(output)
        self.update()

    def update(self):
        self.iteration += 1
        if self.iteration > 400:
            self.reset()
            self.iteration = 0


class PIDController:
    def __init__(self, kp=100.0, ki=0.01, kd=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


if __name__ == "__main__":
    rospy.init_node("pid_control")
    pid_control = PIDControl()
    rospy.spin()
