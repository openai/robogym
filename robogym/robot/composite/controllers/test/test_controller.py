from robogym.envs.rearrange.blocks import make_env
from robogym.robot.composite.controllers.ur_gripper_arm import URGripperArmController


def test_sim_controller_status():
    env = make_env()
    controller = URGripperArmController(env)
    assert controller.get_gripper_regrasp_status() is None


def test_sim_controller_with_regrasp():
    env = make_env(
        parameters=dict(robot_control_params=dict(enable_gripper_regrasp=True,))
    )
    controller = URGripperArmController(env)
    assert controller.get_gripper_regrasp_status() is False
