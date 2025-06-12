import mujoco as m 
import matplotlib.pyplot as plt
import numpy as np
from mujoco import viewer
import os
import mink 
from loop_rate_limiters import RateLimiter

xml = """
<mujoco>
    <option timestep=".0001">
        <flag energy="enable" contact="enable"/>
    </option>
    <default>
        <joint type="hinge" axis="0 -1 0" limited="true" range="-1.14 1.14"/>
        <geom type="capsule" size=".06"/>
    </default>




    <worldbody>
    <body name="arm_handle" mocap="true" pos="0.1 -0.6 0"/>
    <body name="arm" pos="0.1 -.6 0">
        <joint name="base" type="slide" axis="0 0 1" limited="true" range="0 .7"/>
        <geom fromto="0 0 0 0 0 .3" rgba="1 0.5 0 1"/>

        <body name="0" pos="0 0 0">
            <joint name="0" type="hinge" axis="0 0 1" limited="true" range="-3.14 1.14"/>
                <geom fromto="0 0 0 0 0 .3" rgba="1 0.5 0 1"/>
            <body name="1" pos="0 0 .3">
                <joint name="1"/>
                <geom fromto="0 0 .05 0 0 .3" rgba="0.5 0 0 1"/>


                <body name="2" pos="0 0 .3">
                    <joint name="2"/>
                    <geom fromto="0 0 .05 0 0 .3" rgba="0 0.5 0 1"/>

                    <body name="3" pos="0 0 .3">
                        <joint name="3"/>
                        <geom fromto="0 0 .05 0 0 .3" rgba="0 0 0.5 1"/>
                        <site pos="0 0 0.045" name="ee"/>
                    </body>
                </body>
            </body>
        </body>
    </body>

    <body name="microwave" pos="0 0 0">
        <geom type='box' size='.176 .215 .137' pos='0.5 0 0.5' rgba="1 1 1 1"/>
    </body>

    <body name="assembly line" pos="0 0 0">
        <geom type='box' size='.3 1 .3' pos='0.5 0 .15' rgba="1 1 0.5 1"/>
    </body>

    <body name="pallet" pos="0 0 0">
        <geom type='box' size='.5 .5 .05' pos='-0.5 0 0' rgba="1 1 0 1"/>
    </body>

    </worldbody>
    <equality>
    <weld body1="arm" body2="arm_handle" solimp="0.9 0.95 0.001" solref="0.02 1" anchor="0.0 0.1035 0.0"></weld>
    </equality>

    <actuator>
      <motor joint="base" gear="500"/>
      <motor joint="0"    gear="10000"/>
      <motor joint="1"    gear="10000"/>
      <motor joint="2"    gear="10000"/>
      <motor joint="3"    gear="10000"/>
    </actuator>

</mujoco>
"""


pick_state="-2.1611 0.680977 1.15222 0.294862"

place_state="-0.581378 0.856388 1.09171 0.767273"


model = m.MjModel.from_xml_string(xml)
data = m.MjData(model)


# model.reset(step=True)
# init_qpos = model.get_qpos_joints(joint_names=model.rev_joint_names)
# sliders = MultiSliderClass(
#     n_slider = m.n_rev_joint,
#     title = "Slider",
#     slider_vals = init_qpos,
# )



tasks = [ 
    ee_task := mink.FrameTask(
        frame_name="ee",
        frame_type="site",
        position_cost=2.0,
        orientation_cost=1.0,
    ),
    # posture_task := mink.PostureTask(model=model, cost=1e-2)
]

solver = "daqp"
pos_thresh = 1e-4
ori_thresh = 1e-4
max_iters = 500

des = np.array([0.3, 0.3, 0.5])
dy_des = np.zeros(3)

mid = model.body("arm").mocapid[0]

configuration = mink.Configuration(model)

with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:

    rate = RateLimiter(frequency=60.0, warn=False)
    t = 0.0

    mink.move_mocap_to_frame(
        model, data, "arm_handle", "ee", "site"
    )

    while v.is_running():
        T_wt = mink.SE3.from_mocap_name(model, data, "arm_handle")
        ee_task.set_target(T_wt)

        for i in range(max_iters):
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-4, False 
            )
            configuration.integrate_inplace(vel, rate.dt)
            err = ee_task.compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_thresh
            ori_achieved = np.linalg.norm(err[:3]) <= ori_thresh
            if pos_achieved and ori_achieved:
                break

        data.ctrl = configuration.q 
        m.mj_step(model,data)
        m.mj_camlight(model, data)
        v.sync()
        rate.sleep()
        # m.mj_step(model, data)


