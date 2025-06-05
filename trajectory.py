#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import numpy as np
from scipy.interpolate import make_interp_spline
import time
import math
from plotting import plot_system


# --- PyBullet Setup ---
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

robotId = p.loadURDF(
    "./arm/urdf/arm.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=True
)


# --- Joint Info ---
num_joints = p.getNumJoints(robotId)
joint_indices = [i for i in range(num_joints)]

def plan_segment(start, end):
    direction = end - start
    distance = np.linalg.norm(direction)
    direction /= distance

    steps = int(distance / (velocity * dt))
    step_array = np.linspace(0, distance, steps)
    segment = start[None, :] + direction[None, :] * step_array[:, None]
    return segment.tolist()

waypoints = np.array([
    [1, 0, 0.27],
    [0.5, 0.7, 1],
    [-1, 0.6, 2],
])

effector_link_index = num_joints - 1  # assuming last link is the gripper

def pickBox(boxId):
    cube_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    boxCID = p.createConstraint(robotId, effector_link_index, boxId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0], childFrameOrientation=cube_orn)
    p.setCollisionFilterPair(robotId, boxId, effector_link_index, -1, enableCollision=0)

    return boxCID

boxId = p.loadURDF("box.urdf", waypoints[0])
# waypoints *= 0.5
steps_per_second = 500

dt = 1.0 / steps_per_second
velocity = 1

p.setTimeStep(dt)

full_trajectory = []

orientation = p.getQuaternionFromEuler([0, 0, math.pi])
initial_angles = p.calculateInverseKinematics(
    bodyUniqueId=robotId,
    endEffectorLinkIndex=num_joints-1,
    targetPosition=waypoints[0],
)

p.setJointMotorControlArray(
    robotId,
    joint_indices,
    p.POSITION_CONTROL,
    targetPositions=initial_angles
)

for _ in range(5*int(steps_per_second)):  # let it settle
    p.stepSimulation()

pickBox(boxId)
states = []

# --- Execute Trajectory ---
for i in range(len(waypoints)-1):
    segment_trajectory = plan_segment(waypoints[i], waypoints[i+1])

    p.addUserDebugLine(waypoints[i], waypoints[i] + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)
    p.addUserDebugLine(waypoints[i+1], waypoints[i+1] + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)


    prev_point = waypoints[i]

    for point in segment_trajectory:

        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=robotId,
            endEffectorLinkIndex=effector_link_index,
            targetPosition=point,
            maxNumIterations=1000,
            residualThreshold=1e-4
        )

        p.setJointMotorControlArray(
            robotId,
            joint_indices,
            p.POSITION_CONTROL,
            targetPositions=joint_angles
        )

        p.stepSimulation()
        p.addUserDebugLine(prev_point, point, [1, 0, 0], lineWidth=2, lifeTime=0)
        states.append(p.getJointStates(robotId, joint_indices))


positions = np.array([[s[i][0] for i in range(len(s))] for s in states])  # shape (timesteps, joints)
velocities = np.array([[s[i][1] for i in range(len(s))] for s in states])
torques = np.array([[s[i][3] for i in range(len(s))] for s in states])
time_axis = np.arange(len(positions)) * dt
accelerations = np.gradient(velocities, dt, axis=0)

motor_torques = []
for i in range(len(positions)):
    q = positions[i].tolist()
    q_dot = velocities[i].tolist()
    q_ddot = accelerations[i].tolist()
    tau = p.calculateInverseDynamics(robotId, q, q_dot, q_ddot)
    motor_torques.append(tau)

motor_torques = np.array(motor_torques)



plot_system(positions, velocities, accelerations, torques, motor_torques, time_axis, dt, num_joints)
