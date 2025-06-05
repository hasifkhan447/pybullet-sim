#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import numpy as np
from scipy.interpolate import make_interp_spline
import time
from plotting import plot_system

from scipy.signal import butter, filtfilt





# --- PyBullet Setup ---
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
robotId = p.loadURDF("./arm/urdf/arm.urdf", useFixedBase=True)


boxId = p.loadURDF("box.urdf", [1, 0, 0.3])

# --- Joint Info ---
num_joints = p.getNumJoints(robotId)
joint_indices = [i for i in range(num_joints)]

def quintic_interp(p0, p1, v0, v1, a0, a1, T, steps):
    t = np.linspace(0, T, steps)
    traj = np.zeros((steps, 3))
    for i in range(3):  # For x, y, z
        a = p0[i]
        b = v0[i]
        c = a0[i] / 2
        d = (20*p1[i] - 20*p0[i] - (8*v1[i] + 12*v0[i])*T - (3*a0[i] - a1[i])*T**2) / (2*T**3)
        e = (30*p0[i] - 30*p1[i] + (14*v1[i] + 16*v0[i])*T + (3*a0[i] - 2*a1[i])*T**2) / (2*T**4)
        f = (12*p1[i] - 12*p0[i] - (6*v1[i] + 6*v0[i])*T - (a0[i] - a1[i])*T**2) / (2*T**5)
        traj[:, i] = a + b*t + c*t**2 + d*t**3 + e*t**4 + f*t**5
    return traj


def lowpass_filter(data, cutoff=5, fs=240, order=2):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)



def plan_segment(start, end):
    direction = end - start
    distance = np.linalg.norm(direction)
    direction /= distance

    steps = int(distance / (velocity * dt))
    step_array = np.linspace(0, distance, steps)
    segment = start[None, :] + direction[None, :] * step_array[:, None]
    return segment.tolist()

# for i in range (len(waypoints)-1):
#     start = np.array(waypoints[i])
#     end = np.array(waypoints[i + 1])
#     direction = end - start
#     distance = np.linalg.norm(direction)
#     direction /= distance
#
#     steps = int(distance / (velocity * dt))
#     for step in range(steps):
#         pos = start + direction * (velocity * dt * step)
#         full_trajectory.append(pos.tolist())
#



# --- Define 3 Cartesian waypoints ---
waypoints = np.array([
    [1, 0, 0.5],
    [0.5, 0.7, 1],
    [-1, 0.6, 2],
])

# waypoints *= 0.5
duration_per_segment = 1.5  # seconds per segment
steps_per_second = 240
steps = int(duration_per_segment * steps_per_second)

dt = 1.0 / steps_per_second
velocity = 1

def get_boundary_conditions(i, waypoints):
    return np.zeros(3), np.zeros(3)
    if i == 0:
        return np.zeros(3), np.zeros(3)  # start
    elif i == len(waypoints) - 1:
        return np.zeros(3), np.zeros(3)  # end
    else:
        # Estimate velocity and acceleration using finite differences
        v = (waypoints[i+1] - waypoints[i-1]) / (2 * duration_per_segment)
        a = (waypoints[i+1] - 2*waypoints[i] + waypoints[i-1]) / (duration_per_segment**2)
        return v, a

full_trajectory = []

# for i in range(len(waypoints) - 1):
#     p0 = waypoints[i]
#     p1 = waypoints[i+1]
#     v0, a0 = get_boundary_conditions(i, waypoints)
#     v1, a1 = get_boundary_conditions(i+1, waypoints)
#     traj = quintic_interp(
#         p0, p1,
#         v0=v0, v1=v1,
#         a0=a0, a1=a1,
#         T=duration_per_segment,
#         steps=steps
#     )
#     full_trajectory.append(traj)

# --- Generate trajectory ---
# for i in range(len(waypoints) - 1):

# full_trajectory.append(waypoints[-1])

# trajectory = np.vstack(full_trajectory)  # shape: (steps * (n-1), 3)
# trajectory_filtered = lowpass_filter(trajectory, cutoff=5, fs=steps_per_second)
# trajecotry = trajectory_filtered


# for point in waypoints:
#     p.addUserDebugLine(point, point + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)
#
#
# # --- Visualize Trajectory as Arc ---
# for i in range(len(trajectory) - 1):
#     p.addUserDebugLine(trajectory[i], trajectory[i + 1], [1, 0, 0], lineWidth=2, lifeTime=0)

# initial_angles = p.calculateInverseKinematics(robotId, num_joints-1, trajectory[0])
#
# p.setJointMotorControlArray(
#     robotId, joint_indices, p.POSITION_CONTROL, targetPositions=initial_angles,
# )


initial_angles = p.calculateInverseKinematics(robotId, num_joints-1, waypoints[0])

p.setJointMotorControlArray(
    robotId,
    joint_indices,
    p.POSITION_CONTROL,
    targetPositions=initial_angles
)

for _ in range(int(0.4 *steps_per_second)):  # let it settle
    p.stepSimulation()
    time.sleep(dt)


states = []

# --- Execute Trajectory ---
for i in range(len(waypoints)-1):
    segment_trajectory = plan_segment(waypoints[i], waypoints[i+1])

    p.addUserDebugLine(waypoints[i], waypoints[i] + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)
    p.addUserDebugLine(waypoints[i+1], waypoints[i+1] + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)


    prev_point = waypoints[i]


    for point in segment_trajectory:
        joint_angles = p.calculateInverseKinematics(
            robotId,
            num_joints - 1,
            point,
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
        time.sleep(dt)


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
