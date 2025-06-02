#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import numpy as np
from scipy.interpolate import make_interp_spline
import time
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt





# --- PyBullet Setup ---
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
robotId = p.loadURDF("/home/hak/External/Games/shared-win-folder/final_arm_material/urdf/final_arm_material.urdf", useFixedBase=True)

# --- Joint Info ---
num_joints = p.getNumJoints(robotId)
joint_indices = [i for i in range(num_joints)]


# --- Define 2 Poses (start and end) in Cartesian ---
# waypoints = np.array([
#     [1, 0, 0.5],
#     [0.5, 0.7, 0],
#     [-1, 0.6, 1],
# ])
# print(waypoints)
#
#
#
#
#
# t = np.linspace(0,1,len(waypoints))
#
# duration = 1.0  # seconds
# steps_per_second = 240
# total_steps = int(duration * steps_per_second)
# spline = make_interp_spline(t, waypoints, k=2, axis=0)
#
# times = np.linspace(0, 1, total_steps)
# trajectory = spline(times)

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



# --- Define 3 Cartesian waypoints ---
waypoints = np.array([
    [1, 0, 0.5],
    [0.5, 0.7, 1],
    [-1, 0.6, 2],
])

duration_per_segment = 1  # seconds per segment
steps_per_second = 240
steps = int(duration_per_segment * steps_per_second)

def get_boundary_conditions(i, waypoints):
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

for i in range(len(waypoints) - 1):
    p0 = waypoints[i]
    p1 = waypoints[i+1]
    v0, a0 = get_boundary_conditions(i, waypoints)
    v1, a1 = get_boundary_conditions(i+1, waypoints)
    traj = quintic_interp(
        p0, p1,
        v0=v0, v1=v1,
        a0=a0, a1=a1,
        T=duration_per_segment,
        steps=steps
    )
    full_trajectory.append(traj)






trajectory = np.vstack(full_trajectory)  # shape: (steps * (n-1), 3)
trajectory_filtered = lowpass_filter(trajectory, cutoff=5, fs=steps_per_second)
trajecotry = trajectory_filtered


for point in waypoints:
    p.addUserDebugLine(point, point + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)


# --- Visualize Trajectory as Arc ---
for i in range(len(trajectory) - 1):
    p.addUserDebugLine(trajectory[i], trajectory[i + 1], [1, 0, 0], lineWidth=2, lifeTime=0)

states = []
dt = 1.0 / steps_per_second
initial_angles = p.calculateInverseKinematics(robotId, num_joints-1, trajectory[0])

kp = 0.5
kd = 0.5

p.setJointMotorControlArray(
    robotId, joint_indices, p.POSITION_CONTROL, targetPositions=initial_angles,
    positionGains=[kp]*len(joint_indices),
    velocityGains=[kd]*len(joint_indices),
)

for _ in range(int(0.4 *steps_per_second)):  # let it settle
    p.stepSimulation()
    time.sleep(dt)


# --- Execute Trajectory ---
for pos in trajectory:
    # lower_limits = [-3.14] * num_joints
    # upper_limits = [3.14] * num_joints
    # joint_ranges = [6.28] * num_joints
    # rest_poses   = [1.0] * (num_joints - 1) + [1.0]  # encourage last joint movement

    joint_angles = p.calculateInverseKinematics(
        robotId,
        num_joints - 1,
        pos,
        maxNumIterations=400,
        residualThreshold=1e-4
    )

    p.setJointMotorControlArray(
        robotId,
        joint_indices,
        p.POSITION_CONTROL,
        targetPositions=joint_angles,
        positionGains=[kp]*len(joint_indices),
        velocityGains=[kd]*len(joint_indices),
    )

    p.stepSimulation()
    states.append(p.getJointStates(robotId, joint_indices))
    time.sleep(dt)


positions = np.array([[s[i][0] for i in range(len(s))] for s in states])  # shape (timesteps, joints)
velocities = np.array([[s[i][1] for i in range(len(s))] for s in states])
torques = np.array([[s[i][3] for i in range(len(s))] for s in states])


time_axis = np.arange(len(positions)) * dt

plt.figure(figsize=(10, 8))
plt.subplot(5, 1, 1)
for joint_id in range(num_joints):
        plt.plot(time_axis, positions[:, joint_id], label=f"joint {joint_id}")
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)
plt.title(f'Joint Positions')

plt.subplot(5, 1, 2)
for joint_id in range(num_joints):
        plt.plot(time_axis, velocities[:, joint_id], label=f"joint {joint_id}")
        plt.title(f'Joint {joint_id} Velocity')
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)
plt.title(f'Joint Velocities')

plt.subplot(5, 1, 3)
for joint_id in range(num_joints):
        accelerations = np.gradient(velocities, dt, axis=0)
        plt.plot(time_axis, accelerations[:, joint_id], label=f"joint {joint_id}")
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)
plt.title(f'Joint Accelerations')

plt.subplot(5, 1, 4)
for joint_id in range(num_joints):
        plt.plot(time_axis, torques[:, joint_id], label=f"joint {joint_id}")
plt.title(f'Joint Torques')
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)

motor_torques = []
for i in range(len(positions)):
    q = positions[i].tolist()
    q_dot = velocities[i].tolist()
    q_ddot = accelerations[i].tolist()
    tau = p.calculateInverseDynamics(robotId, q, q_dot, q_ddot)
    motor_torques.append(tau)

motor_torques = np.array(motor_torques)

# --- Plot Motor Torques ---
plt.subplot(5, 1, 5)
for joint_id in range(num_joints):
    plt.plot(time_axis, motor_torques[:, joint_id], label=f"joint {joint_id}")
plt.title("Motor Torques (Inverse Dynamics)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.05))
plt.grid(True)



plt.tight_layout()
plt.show()


# --- Torque Statistics ---
for joint_id in range(num_joints):
    joint_torques = motor_torques[:, joint_id]
    avg_torque = np.mean(np.abs(joint_torques))
    max_torque = np.max(np.abs(joint_torques))

    joint_velocities = velocities[:, joint_id]
    avg_velocity = np.mean(np.abs(joint_velocities))
    max_velocity = np.max(np.abs(joint_velocities))

    print(f"Joint {joint_id}:")
    print(f"  Avg Torque     = {avg_torque:.4f}  Avg velocity   = {avg_velocity:.4f}")
    print(f"  Max Torque     = {max_torque:.4f}  Max Velocity   = {max_velocity:.4f}")
