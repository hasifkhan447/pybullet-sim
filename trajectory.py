#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import numpy as np
from scipy.interpolate import make_interp_spline
import time
import math
from plotting import plot_system
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np



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
    # baseOrientation=[0, 0, 0, 1],
    useFixedBase=True
)


steps_per_second = 500

dt = 1.0 / steps_per_second
velocity = 1

p.setTimeStep(dt)


# --- Joint Info ---
num_joints = p.getNumJoints(robotId)
joint_indices = [i for i in range(num_joints)]

def plan_segment_quintic(start, end, duration=1.0, dt=dt):
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

def plan_segment_quintic_with_orientation(start_pos, end_pos, start_quat, end_quat, duration=1.0, dt=dt):
    t = np.arange(0, duration + dt, dt)
    T = duration

    # Quintic polynomial coefficients (same as yours)
    a0, a1, a2 = 0, 0, 0
    a3 = 10 / T**3
    a4 = -15 / T**4
    a5 = 6 / T**5

    s = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5  # scalar 0 to 1

    start_pos, end_pos = np.array(start_pos), np.array(end_pos)
    segment = start_pos + (end_pos - start_pos) * s[:, None]

    # Orientation interpolation using Slerp
    key_times = [0, duration]
    rotations = R.from_quat([start_quat, end_quat])
    slerp = Slerp(key_times, rotations)
    orientations = slerp(s * duration).as_quat()

    return segment.tolist(), orientations.tolist()





def go_through_waypoints(waypoints, pick_orientation, place_orientation):

    for i in range(len(waypoints)):
        p.addUserDebugLine(waypoints[i], waypoints[i] + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)


    for i in range(len(waypoints)-1):
        # segment_trajectory = plan_segment_constant_velocity(waypoints[i], waypoints[i+1])
        # segment_trajectory = plan_segment_quintic(waypoints[i], waypoints[i+1], duration=1)

        if i == 0:
            print("Rotating rq")
            segment_trajectory, orientations = plan_segment_quintic_with_orientation(waypoints[i], waypoints[i+1], pick_orientation, place_orientation, duration=1)
        else:
            segment_trajectory, orientations = plan_segment_quintic_with_orientation(waypoints[i], waypoints[i+1], place_orientation, place_orientation, duration=1)



        prev_point = waypoints[i]
        for point_idx in range(len(segment_trajectory)):
            if point_idx % 10 == 0:
                point = segment_trajectory[point_idx]
                p.addUserDebugLine(prev_point, point, [1, 0, 0], lineWidth=2, lifeTime=0)
                prev_point = point

        for idx in range(len(segment_trajectory)):
            joint_angles = p.calculateInverseKinematics(
                bodyUniqueId=robotId,
                endEffectorLinkIndex=effector_link_index,
                targetPosition=segment_trajectory[idx],
                targetOrientation=orientations[idx],
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
        parentFramePosition=[0, 0.15, 0],
        childFramePosition=[0, 0, 0],
        childFrameOrientation=p.getQuaternionFromEuler([-math.pi/2,0,0])
    )

    p.setCollisionFilterPair(robotId, boxId, effector_link_index, -1, enableCollision=0)
    p.setCollisionFilterPair(robotId, boxId, effector_link_index-1, -1, enableCollision=0)

    for i in range(int(steps_per_second)):
        p.stepSimulation()
    return boxCID




def dropBox(boxCID):
    p.removeConstraint(boxCID)
    for i in range(int(steps_per_second)):
        p.stepSimulation()



full_trajectory = []

box_x_len = 0.43 + 0.05
box_y_len = 0.35 + 0.05
box_z_len = 0.28 


waypoints = np.array([
    [0.7, 0.5, 0.5],
    [-0.4, 0.8, 2],
    [-1.2, 0.7, box_z_len+ 0.04],
])

normal_waypoint_3 = [-0.4, 0.8,2]
alternative_waypoint_3 = [-0.8, 1.4, 2]

pick_orientation = p.getQuaternionFromEuler([-math.pi/2,0,0])
initial_angles = p.calculateInverseKinematics(
    bodyUniqueId=robotId,
    endEffectorLinkIndex=effector_link_index,
    targetPosition=waypoints[0],
    targetOrientation=pick_orientation
)

p.setJointMotorControlArray(
    robotId,
    joint_indices,
    p.POSITION_CONTROL,
    targetPositions=initial_angles
)


states = []


def palletize(waypoints, pick_orientation, place_orientation):
    boxId = p.loadURDF("box.urdf", [0.7,0.5,0.3])

    for i in range(2*int(steps_per_second)):
        p.stepSimulation()


    boxCID = pickBox(boxId)
    go_through_waypoints(waypoints, pick_orientation, place_orientation)
    dropBox(boxCID)
    go_through_waypoints(waypoints[::-1], place_orientation, pick_orientation)


num_stacks = 6
for i in range(num_stacks):
# Bottom-right
    waypoints[-1][0] += box_x_len
    waypoints[-2] = normal_waypoint_3
    waypoints[-2][0] += box_x_len - 0.4
    waypoints[-2][2] = waypoints[-1][2] + 0.7
    place_orientation = p.getQuaternionFromEuler([-np.pi/2, 0, np.pi/2])
    palletize(waypoints, pick_orientation, place_orientation)

# Bottom-left
    waypoints[-1][0] -= box_x_len
    waypoints[-2] = normal_waypoint_3
    waypoints[-2][0] -= box_x_len + 0.4
    waypoints[-2][2] = waypoints[-1][2] + 0.7
    place_orientation = p.getQuaternionFromEuler([-np.pi/2, 0, 0])
    palletize(waypoints, pick_orientation, place_orientation)

# Top-left
    waypoints[-1][1] += box_y_len
    waypoints[-2] = alternative_waypoint_3
    waypoints[-2][1] += box_y_len - 0.4
    waypoints[-2][2] = waypoints[-1][2] + 0.7
    place_orientation = p.getQuaternionFromEuler([-np.pi/2, 0, -np.pi/2])
    palletize(waypoints, pick_orientation, place_orientation)

# Top-right
    waypoints[-1][0] += box_x_len
    waypoints[-2] = alternative_waypoint_3
    waypoints[-2][0] += box_x_len - 0.4
    waypoints[-2][2] = waypoints[-1][2] + 0.7
    place_orientation = p.getQuaternionFromEuler([-np.pi/2, 0, -np.pi])
    palletize(waypoints, pick_orientation, place_orientation)

# Move up for next layer
    waypoints[-1][0] -= box_x_len
    waypoints[-1][1] -= box_y_len
    waypoints[-1][2] += box_z_len







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
