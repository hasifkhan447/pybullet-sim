#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import numpy as np
from scipy.interpolate import make_interp_spline
import time
import matplotlib.pyplot as plt


# --- PyBullet Setup ---
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
robotId = p.loadURDF("/home/hak/External/Games/shared-win-folder/final_arm/urdf/final_arm.urdf", useFixedBase=True)

# --- Joint Info ---
num_joints = p.getNumJoints(robotId)
joint_indices = [i for i in range(num_joints)]


# --- Define 2 Poses (start and end) in Cartesian ---
waypoints = np.array([
    [1, 0, 0.3],
    [0, 1, 0.3],
    [0, 0, 1],
    [1, 0.5, 0.3],
    [0, 0.5, 1]
])
print(waypoints)
t = np.linspace(0,1,len(waypoints))





# --- B-Spline ---
spline = make_interp_spline(t, waypoints, k=4, axis=0)
times = np.linspace(0, 1, 100)
trajectory = spline(times)



for point in waypoints:
    p.addUserDebugLine(point, point + np.array([0, 0, 0.1]), [0, 1, 0], lineWidth=50, lifeTime=0)


# --- Visualize Trajectory as Arc ---
for i in range(len(trajectory) - 1):
    p.addUserDebugLine(trajectory[i], trajectory[i + 1], [1, 0, 0], lineWidth=2, lifeTime=0)



# --- Record Data ---
recorded_joint_angles = []


states = []

dt = 1.0 / 240.0


for point in waypoints:
    initial_angles = p.calculateInverseKinematics(robotId, num_joints-1, point)
    p.setJointMotorControlArray(
        robotId, joint_indices, p.POSITION_CONTROL, targetPositions=initial_angles,
    )


    for _ in range(500):  # let it settle
        p.stepSimulation()
        time.sleep(dt)


initial_angles = p.calculateInverseKinematics(robotId, num_joints-1, trajectory[0])
p.setJointMotorControlArray(
    robotId, joint_indices, p.POSITION_CONTROL, targetPositions=initial_angles,
)


for _ in range(500):  # let it settle
    p.stepSimulation()
    time.sleep(dt)


# --- Execute Trajectory ---
for pos in trajectory:
    joint_angles = p.calculateInverseKinematics(robotId, num_joints-1, pos)

    p.setJointMotorControlArray(
        robotId,
        joint_indices,
        p.POSITION_CONTROL,
        targetPositions=joint_angles
    )

    p.stepSimulation()


    recorded_joint_angles.append(joint_angles)
    states.append(p.getJointStates(robotId, joint_indices))

    time.sleep(dt)


positions = np.array([[s[i][0] for i in range(len(s))] for s in states])  # shape (timesteps, joints)
velocities = np.array([[s[i][1] for i in range(len(s))] for s in states])
torques = np.array([[s[i][3] for i in range(len(s))] for s in states])




joint_id = 0
time_axis = np.arange(len(positions)) * dt

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
for joint_id in range(num_joints):
        plt.plot(time_axis, positions[:, joint_id], label=f"joint {joint_id}")
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)
plt.title(f'Joint Positions')

plt.subplot(4, 1, 2)
for joint_id in range(num_joints):
        plt.plot(time_axis, velocities[:, joint_id], label=f"joint {joint_id}")
        plt.title(f'Joint {joint_id} Velocity')
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)
plt.title(f'Joint Velocities')

plt.subplot(4, 1, 3)
for joint_id in range(num_joints):
        accelerations = np.gradient(velocities, dt, axis=0)
        plt.plot(time_axis, accelerations[:, joint_id], label=f"joint {joint_id}")
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)
plt.title(f'Joint Accelerations')



plt.subplot(4, 1, 4)
for joint_id in range(num_joints):
        plt.plot(time_axis, torques[:, joint_id], label=f"joint {joint_id}")
plt.title(f'Joint Torques')
plt.legend(loc='center left',bbox_to_anchor=(1,0.05))
plt.grid(True)

plt.tight_layout()
plt.show()






# `recorded_joint_angles` now holds the full trajectory
print(recorded_joint_angles)
