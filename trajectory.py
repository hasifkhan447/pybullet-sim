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
#
# robotId = p.loadURDF(
#     "./arm/urdf/arm.urdf",
#     basePosition=[0, 0, 0],
#     baseOrientation=[0, 0, 0, 1],
#     useFixedBase=True
# )
#

robotId = p.loadURDF(
    "./larger-arm/urdf/larger-arm.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=True
)



# --- Joint Info ---
num_joints = p.getNumJoints(robotId)
joint_indices = [i for i in range(num_joints)]


def axiscreator(bodyId, linkId = -1):
    # print(f'axis creator at bodyId = {bodyId} and linkId = {linkId} as XYZ->RGB')
    x_axis = p.addUserDebugLine(lineFromXYZ = [0, 0, 0] ,
                                lineToXYZ = [0.1, 0, 0],
                                lineColorRGB = [1, 0, 0] ,
                                lineWidth = 0.1 ,
                                lifeTime = 0 ,
                                parentObjectUniqueId = bodyId ,
                                parentLinkIndex = linkId )

    y_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0]  ,
                                lineToXYZ            = [0, 0.1, 0],
                                lineColorRGB         = [0, 1, 0]  ,
                                lineWidth            = 0.1        ,
                                lifeTime             = 0          ,
                                parentObjectUniqueId = bodyId     ,
                                parentLinkIndex      = linkId     )

    z_axis = p.addUserDebugLine(lineFromXYZ          = [0, 0, 0]  ,
                                lineToXYZ            = [0, 0, 0.1],
                                lineColorRGB         = [0, 0, 1]  ,
                                lineWidth            = 0.1        ,
                                lifeTime             = 0          ,
                                parentObjectUniqueId = bodyId     ,
                                parentLinkIndex      = linkId     )
    return [x_axis, y_axis, z_axis]







def plan_segment_constant_velocity(start, end):
    direction = end - start
    distance = np.linalg.norm(direction)
    direction /= distance

    steps = int(distance / (velocity * dt))
    step_array = np.linspace(0, distance, steps)
    segment = start[None, :] + direction[None, :] * step_array[:, None]
    return segment.tolist()




def plan_segment_quintic(start, end, duration=1.0, dt=0.01):
    t = np.arange(0, duration + dt, dt)
    T = duration

    # Quintic polynomial coefficients for scalar trajectory from 0 to 1 with zero start/end vel & acc
    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 10 / T**3
    a4 = -15 / T**4
    a5 = 6 / T**5

    s = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5  # scalar position from 0 to 1

    start, end = np.array(start), np.array(end)
    segment = start + (end - start) * s[:, None]

    return segment.tolist()


waypoints = np.array([
    [0.7, 0, 0.5],
    [0.3, 1, 1],
    [-1.2, 0.7, 1.6],
])


effector_link_index = num_joints - 1  # assuming last link is the gripper

def pickBox(boxId):
    _, eff_orn = p.getLinkState(robotId, effector_link_index)[:2]
    boxCID = p.createConstraint(
        parentBodyUniqueId=robotId,
        parentLinkIndex=effector_link_index,
        childBodyUniqueId=boxId,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0.05, 0],
        childFramePosition=[0, 0, 0],
        childFrameOrientation=p.getQuaternionFromEuler([math.pi/2,0,0])
    )


    p.setCollisionFilterPair(robotId, boxId, effector_link_index, -1, enableCollision=0)

    return boxCID

# waypoints *= 0.5
steps_per_second = 500

dt = 1.0 / steps_per_second
velocity = 1

p.setTimeStep(dt)

full_trajectory = []


boxId = p.loadURDF("box.urdf", waypoints[0])

orientation = p.getQuaternionFromEuler([-math.pi/2,0,0])
initial_angles = p.calculateInverseKinematics(
    bodyUniqueId=robotId,
    endEffectorLinkIndex=effector_link_index,
    targetPosition=waypoints[0],
    targetOrientation=orientation
)

p.setJointMotorControlArray(
    robotId,
    joint_indices,
    p.POSITION_CONTROL,
    targetPositions=initial_angles
)

pickBox(boxId)

for i in range(int(steps_per_second)):
    p.stepSimulation()

states = []

for i in range(len(waypoints)-1):
    p.addUserDebugLine(waypoints[i], waypoints[i] + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)



# --- Execute Trajectory ---
for i in range(len(waypoints)-1):
    segment_trajectory = plan_segment_constant_velocity(waypoints[i], waypoints[i+1])
    # segment_trajectory = plan_segment_quintic(waypoints[i], waypoints[i+1])


    prev_point = waypoints[i]
    for point_idx in range(len(segment_trajectory)):
        if point_idx % 10 == 0:
            point = segment_trajectory[point_idx]
            p.addUserDebugLine(prev_point, point, [1, 0, 0], lineWidth=2, lifeTime=0)
            prev_point = point

    for point in segment_trajectory:
        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=robotId,
            endEffectorLinkIndex=effector_link_index,
            targetPosition=point,
            targetOrientation=orientation,
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
        time.sleep(dt)
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
