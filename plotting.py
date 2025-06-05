import matplotlib.pyplot as plt
import pybullet as p
import numpy as np
from scipy.signal import butter, filtfilt
def plot_system(positions, velocities, accelerations, torques, motor_torques, time_axis, dt, num_joints):
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


# --- Plot Motor Torques ---
    plt.subplot(5, 1, 5)
    for joint_id in range(num_joints):
        plt.plot(time_axis, motor_torques[:, joint_id], label=f"joint {joint_id}")
    plt.title("Motor Torques (Inverse Dynamics)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.05))
    plt.grid(True)



    plt.tight_layout()
    plt.show()

