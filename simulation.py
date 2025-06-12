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
        <geom type="cylinder" size=".06"/>
    </default>
    <worldbody>
    <body name="arm" pos="0 0 .5">
        <joint name="root" type="hinge" axis="0 0 1" limited="true" range="-3.14 1.14"/>
            <geom fromto="0 0 0 0 0 .3" rgba="1 1 0 1"/>
        <body name="1" pos="0 0 .3">
            <joint name="1"/>
            <geom fromto="0 0 .01 0 0 .3" rgba="1 0 0 1"/>


            <body name="2" pos="0 0 .3">
                <joint name="2"/>
                <geom fromto="0 0 .01 0 0 .3" rgba="0 1 0 1"/>

                <body name="3" pos="0 0 .3">
                    <joint name="3"/>
                    <geom fromto="0 0 .01 0 0 .3" rgba="0 0 1 1"/>
                </body>
            </body>
        </body>
    </body>


    <body name="microwave" pos="0 0 0">
        <geom type='box' size='.3556 .4318 .2794' pos='0.5 0 0.5' rgba="1 1 0 1"/>
    </body>


    </worldbody>


    <actuator>
      <motor joint="root" gear="100"/>
      <motor joint="1"    gear="100"/>
      <motor joint="2"    gear="100"/>
      <motor joint="3"    gear="100"/>
    </actuator>

</mujoco>
"""

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
