import mujoco as m 
import matplotlib.pyplot as plt
import numpy as np
from mujoco import viewer
import os

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
    <body name="0" pos="0.1 -.6 .4">
        <joint name="root" type="hinge" axis="0 0 1" limited="true" range="-3.14 1.14"/>
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


    <actuator>
      <motor joint="root" gear="10000"/>
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




with viewer.launch_passive(model, data) as v:
    while v.is_running():
        m.mj_step(model, data)
        v.sync()
