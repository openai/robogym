import random

import numpy as np
from mujoco_py import cymj, functions
from numpy.random.mtrand import _rand as global_randstate

from robogym.mujoco.forward_kinematics import ForwardKinematics
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.utils.rotation import uniform_quat

XML_BALL = """
<mujoco>
  <worldbody>
    <body name="ball">
      <freejoint name="ball_joint"/>
      <geom  name="sphere"    pos="0.00 0.00 0.00"  type="sphere" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


XML_ARM = """
<mujoco>
  <worldbody>
    <body name="arm">
      <joint type="hinge" name="hinge_joint" axis="0 0 1"/>
      <geom  name="sphere"    pos="0.00 0.00 0.00"  type="sphere" size="0.1 0.1 0.1"/>
      <body name="forearm" pos="1 0 0">
        <joint type="slide" axis="1 0 0" name="slide_joint"/>
        <geom  name="box"   pos="0.00 0.00 0.00"  type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_simple_mujoco_setup():
    ball_one = (
        MujocoXML.from_string(XML_BALL)
        .add_name_prefix("ball_one:")
        .set_named_objects_attr("ball_one:ball", pos=[1, 0, 0])
    )

    ball_two = (
        MujocoXML.from_string(XML_BALL)
        .add_name_prefix("ball_two:")
        .set_named_objects_attr("ball_two:ball", pos=[-1, 0, 0])
    )

    main = (
        MujocoXML().add_default_compiler_directive().append(ball_one).append(ball_two)
    )

    simulation = SimulationInterface(main.build())

    simulation.register_joint_group("ball_one", "ball_one:ball_joint")
    simulation.register_joint_group("ball_two", "ball_two:ball_joint")

    assert simulation.get_qpos("ball_one").shape == (7,)
    assert simulation.get_qpos("ball_two").shape == (7,)

    assert simulation.get_qvel("ball_one").shape == (6,)
    assert simulation.get_qvel("ball_two").shape == (6,)

    qpos1 = np.random.randn(3)
    qrot1 = uniform_quat(global_randstate)
    qpos1_combined = np.concatenate([qpos1, qrot1])

    qpos2 = np.random.randn(3)
    qrot2 = uniform_quat(global_randstate)
    qpos2_combined = np.concatenate([qpos2, qrot2])

    simulation.set_qpos("ball_one", qpos1_combined)
    simulation.set_qpos("ball_two", qpos2_combined)

    assert np.linalg.norm(simulation.get_qpos("ball_one") - qpos1_combined) < 1e-6
    assert np.linalg.norm(simulation.get_qpos("ball_two") - qpos2_combined) < 1e-6


def test_more_complex_mujoco_setup():
    xml = (
        MujocoXML()
        .add_default_compiler_directive()
        .append(
            MujocoXML.from_string(XML_ARM)
            .add_name_prefix("arm_one:")
            .set_named_objects_attr("arm_one:ball", pos=[0, 1, 0])
        )
        .append(
            MujocoXML.from_string(XML_ARM)
            .add_name_prefix("arm_two:")
            .set_named_objects_attr("arm_two:ball", pos=[0, -1, 0])
        )
    )

    simulation = SimulationInterface(xml.build())

    simulation.register_joint_group("arm_one", "arm_one:")
    simulation.register_joint_group("arm_one_hinge", "arm_one:hinge_joint")
    simulation.register_joint_group("arm_two", "arm_two:")
    simulation.register_joint_group("arm_two_hinge", "arm_two:hinge_joint")

    assert simulation.get_qpos("arm_one").shape == (2,)
    assert simulation.get_qvel("arm_one").shape == (2,)
    assert simulation.get_qpos("arm_two").shape == (2,)
    assert simulation.get_qvel("arm_two").shape == (2,)

    assert simulation.get_qpos("arm_one_hinge").shape == (1,)
    assert simulation.get_qvel("arm_one_hinge").shape == (1,)
    assert simulation.get_qpos("arm_two_hinge").shape == (1,)
    assert simulation.get_qvel("arm_two_hinge").shape == (1,)

    initial_qpos_one = simulation.get_qpos("arm_one")
    initial_qpos_two = simulation.get_qpos("arm_two")

    simulation.set_qpos("arm_one_hinge", 0.1)

    # Chech that we are setting the right hinge joint
    assert np.linalg.norm(simulation.get_qpos("arm_one") - initial_qpos_one) > 0.09
    assert np.linalg.norm(simulation.get_qpos("arm_two") - initial_qpos_two) < 1e-6


def test_set_attributes_mixed_precision():
    main = (
        MujocoXML()
        .add_default_compiler_directive()
        .append(
            MujocoXML.from_string(XML_BALL).set_named_objects_attr(
                "ball", pos=[1, 1e-8, 1e-12]
            )
        )
    )

    simulation = SimulationInterface(main.build())

    ball_id = simulation.sim.model.body_name2id("ball")
    ball_pos = simulation.sim.model.body_pos[ball_id]

    target_pos = np.array([1, 1e-8, 1e-12])

    # test relative error cause absolute error can be quite small either way
    assert np.linalg.norm((ball_pos / target_pos) - 1) < 1e-6


def test_forward_kinematics_on_inverted_pendulum():
    mxml = MujocoXML.parse(
        "test/inverted_pendulum/inverted_double_pendulum.xml"
    ).add_name_prefix("ivp:")
    simulation = SimulationInterface(mxml.build())
    simulation.register_joint_group("pendulum", "ivp:")

    joint_names = list(map(lambda x: "ivp:%s" % x, ["hinge", "hinge2"]))

    site_names = list(map(lambda x: "ivp:%s" % x, ["hinge2_site", "tip"]))

    KIN = ForwardKinematics.prepare(
        mxml, "ivp:cart", np.zeros(3), np.zeros(3), site_names, joint_names
    )

    for _ in range(5):
        simulation.mj_sim.data.ctrl[0] = random.random()
        for _ in range(100):
            simulation.step()

        simulation.forward()

        site_positions = np.array(
            [simulation.mj_sim.data.get_site_xpos(site) for site in site_names]
        )

        joint_pos = simulation.get_qpos("pendulum")

        kinemetics_positions = KIN.compute(joint_pos, return_joint_pos=True)

        assert (np.abs(site_positions - kinemetics_positions[:2]) < 1e-6).all()
        assert (np.abs(site_positions[0] - kinemetics_positions[-1]) < 1e-6).all()


def test_remove_elem():
    ball_without_joint = MujocoXML.from_string(XML_BALL).remove_objects_by_tag(
        "freejoint"
    )

    ref_xml = """
<mujoco>
  <worldbody>
    <body name="ball">
      <geom name="sphere" pos="0.00 0.00 0.00" size="0.1 0.1 0.1" type="sphere" />
    </body>
  </worldbody>
</mujoco>
"""
    assert ref_xml.strip() == ball_without_joint.xml_string().strip()


def test_mj_error_callback():
    message = None
    called = False

    def callback(msg):
        nonlocal message
        message = msg.decode()
        raise RuntimeError(message)

    cymj.set_error_callback(callback)

    try:
        with cymj.wrap_mujoco_warning():
            functions.mju_error("error")
    except RuntimeError as e:
        assert e.args[0] == "error"
        assert message == "error"
        called = True

    assert called
