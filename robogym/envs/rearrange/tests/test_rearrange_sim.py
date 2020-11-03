import numpy as np
import pytest

from robogym.envs.rearrange.blocks import make_env as make_blocks_env
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.robot.robot_interface import ControlMode, TcpSolverMode
from robogym.wrappers.util import DiscretizeActionWrapper


def test_sim_sizes():
    env = make_blocks_env(
        parameters=dict(robot_control_params=dict(tcp_solver_mode="mocap"))
    )
    env.reset()
    assert env.sim.model.njmax == 2000
    assert env.sim.model.nconmax == 500
    assert env.sim.model.nuserdata == 2000
    assert env.sim.model.nuser_actuator == 16
    assert env.sim.model.opt.timestep == 0.001

    arm_simulation = env.robot.robots[0].controller_arm.mj_sim
    assert arm_simulation.model.opt.timestep == 0.001
    assert arm_simulation == env.sim


@pytest.mark.parametrize(
    "control_mode,tcp_solver_mode",
    [
        [ControlMode.TCP_WRIST, TcpSolverMode.MOCAP_IK],
        [ControlMode.TCP_ROLL_YAW, TcpSolverMode.MOCAP_IK],
    ],
)
def test_dual_sim_sizes(control_mode, tcp_solver_mode):
    env = make_blocks_env(
        parameters=dict(
            robot_control_params=dict(
                control_mode=control_mode, tcp_solver_mode=tcp_solver_mode
            )
        )
    )
    env.reset()
    assert env.sim.model.njmax == 2000
    assert env.sim.model.nconmax == 500
    assert env.sim.model.nuserdata == 2000
    assert env.sim.model.nuser_actuator == 16

    arm_simulation = env.robot.robots[0].controller_arm.mj_sim

    assert arm_simulation != env.sim
    assert arm_simulation.model.njmax == 200
    assert arm_simulation.model.nconmax == 200
    assert arm_simulation.model.nuserdata == 200
    assert arm_simulation.model.nuser_actuator == 16


def test_dual_sim_stepping_times():
    """Ensures time advances the same way for main and solver sim to avoid
    bugs like double-stepping sims
    """
    env = make_blocks_env(
        parameters=dict(
            robot_control_params=dict(
                control_mode=ControlMode.TCP_ROLL_YAW,
                tcp_solver_mode=TcpSolverMode.MOCAP_IK,
            )
        )
    )
    env.reset()

    def get_main_sim_time():
        return env.mujoco_simulation.mj_sim.data.time

    def get_solver_sim_time():
        return env.robot.robots[0].controller_arm.mj_sim.data.time

    initial_offset = get_main_sim_time() - get_solver_sim_time()
    for _ in range(10):
        env.step(env.action_space.sample())
        time_diff = get_main_sim_time() - get_solver_sim_time()
        assert np.isclose(time_diff, initial_offset, atol=1e-5), (
            f"Time does not advance at the same"
            f"rate for both sims. "
            f"diff: {time_diff - initial_offset:.3f}s"
        )


def test_dual_sim_timesteps():
    env = make_blocks_env(constants=dict(mujoco_timestep=0.008,))
    env.reset()

    arm_simulation = env.robot.robots[0].controller_arm.mj_sim
    assert env.sim.model.opt.timestep == 0.008
    assert arm_simulation.model.opt.timestep == 0.008


def test_dual_sim_gripper_sync():
    env = make_blocks_env(
        parameters=dict(
            n_random_initial_steps=0,
            robot_control_params=dict(tcp_solver_mode=TcpSolverMode.MOCAP_IK,),
        ),
        starting_seed=2,
    )
    env.reset()
    # Remove discretize action wrapper so the env accepts zero action.
    assert isinstance(env, DiscretizeActionWrapper)
    env = env.env

    def _get_helper_gripper_qpos():
        helper_sim = env.robot.robots[0].controller_arm.simulation
        return helper_sim.gripper.observe().joint_positions()[0]

    # verify initial gripper state is synced between two sims
    main_sim_gripper_pos = env.observe()["gripper_qpos"]
    assert np.isclose(main_sim_gripper_pos, 0.0, atol=1e-4)
    assert np.isclose(_get_helper_gripper_qpos(), 0.0, atol=1e-4)

    # open gripper
    zeros_open_gripper = np.zeros_like(env.action_space.sample())
    zeros_open_gripper[-1] = -1
    for i in range(25):
        # first 5 steps deviate more, then the helper arm should catch up.
        obs_tol = 0.012 if i < 5 else 0.001
        env.step(zeros_open_gripper)
        assert np.isclose(
            env.observe()["gripper_qpos"], _get_helper_gripper_qpos(), atol=obs_tol
        )

    # verify final gripper state is synced between two sims
    main_sim_gripper_pos = env.observe()["gripper_qpos"]
    assert np.isclose(main_sim_gripper_pos, -0.04473, atol=1e-4)
    assert np.isclose(_get_helper_gripper_qpos(), -0.04473, atol=1e-4)


@pytest.mark.parametrize(
    "reset_controller_error, max_position_change, expected_displacement, response_time_steps",
    [
        (True, 0.165, 0.036, 5),
        (False, 0.05, 0.0363, 12),
        (True, 0.1, 0.022, 5),
        (False, 0.03, 0.022, 12),
    ],
)
def test_mocap_ik_impulse_response(
    reset_controller_error,
    max_position_change,
    expected_displacement,
    response_time_steps,
):
    """
    Ensures that the expected displacement is achieved after an 'impulse' action at a given TCP dimension to the policy
    within a given responsiveness.
    :param reset_controller_error: Whether the test resets controller error on mocap_ik motion
    :param max_position_change: Max position change in TCP space
    :param expected_displacement: Expected steady-state displacement caused by the impulse action
    :param response_time_steps: Number of steps it should take to reach 90% of the expected steady-state displacement
    """

    def make_env(arm_reset_controller_error, max_position_change):
        env = make_blocks_env(
            parameters=dict(
                n_random_initial_steps=0,
                robot_control_params=dict(
                    control_mode=ControlMode.TCP_ROLL_YAW,
                    tcp_solver_mode=TcpSolverMode.MOCAP_IK,
                    arm_reset_controller_error=arm_reset_controller_error,
                    max_position_change=max_position_change,
                ),
            ),
            starting_seed=0,
        )
        env.reset()
        # Remove discretize action wrapper so the env accepts zero action.
        assert isinstance(env, DiscretizeActionWrapper)
        return env.env

    def get_trajectory(impulse_dim, impulse_lag=2, **kwargs):
        """
        Generates a trajectory with applying an "impulse" maximum action in the
        specified dimension.
        :param impulse_dim: Specified action dimension to apply the impulse to
        :param impulse_lag: Lag in steps before we apply the impulse
        :return: Gripper pos trajectories for main and helper arms
        """
        mocap_env = make_env(**kwargs)
        zero_action = np.zeros(mocap_env.action_space.shape[0])
        impulse_action = zero_action.copy()
        impulse_action[impulse_dim] = 1

        gripper_pos = []
        helper_pos = []
        main_arm = mocap_env.robot.robots[0]
        helper_arm = mocap_env.robot.robots[0].controller_arm

        for _ in range(impulse_lag):
            mocap_env.step(zero_action)
            gripper_pos.append(main_arm.observe().tcp_xyz())
            helper_pos.append(helper_arm.observe().tcp_xyz())

        for _ in range(1):
            mocap_env.step(impulse_action)
            gripper_pos.append(main_arm.observe().tcp_xyz())
            helper_pos.append(helper_arm.observe().tcp_xyz())

        for _ in range(40):
            mocap_env.step(zero_action)
            gripper_pos.append(main_arm.observe().tcp_xyz())
            helper_pos.append(helper_arm.observe().tcp_xyz())

        return np.asarray(gripper_pos), np.asarray(helper_pos)

    impulse_lag = 2
    # apply an impulse action in x, y, z dimensions
    for control_dimension in range(3):
        main_pos, helper_pos = get_trajectory(
            impulse_dim=control_dimension,
            impulse_lag=impulse_lag,
            max_position_change=max_position_change,
            arm_reset_controller_error=reset_controller_error,
        )
        # normalize main pos
        main_pos = main_pos - main_pos[0, :]
        total_displacement = main_pos[-1, control_dimension]
        assert np.isclose(total_displacement, expected_displacement, atol=1e-3)

        # assert that we can get within 10% of the max value in <rise_time_steps>  steps
        assert (
            np.abs(main_pos[impulse_lag + response_time_steps, control_dimension])
            > total_displacement * 0.9
        )


def test_hide_geoms():
    """ Tests all modes of RearrangeSimulationInterface._hide_geoms """
    env = make_blocks_env(constants=dict(vision=True), starting_seed=0)
    env.reset()
    sim: RearrangeSimulationInterface = env.mujoco_simulation

    img_no_hiding = sim.render()
    img_no_hiding2 = sim.render()
    # Sanity check that two renderings of the same state are identical.
    assert np.allclose(img_no_hiding, img_no_hiding2)

    with sim.hide_target():
        img_hide_targets = sim.render()
        assert not np.allclose(
            img_no_hiding, img_hide_targets
        ), "Image with hidden targets should be different than without hiding targets"

    with sim.hide_target(hide_robot=True):
        img_hide_targets_and_robot = sim.render()
        assert not np.allclose(
            img_hide_targets, img_hide_targets_and_robot
        ), "Hiding the robot should result in a different image"

    with sim.hide_objects():
        img_hide_objects = sim.render()
        assert not np.allclose(
            img_hide_objects, img_hide_targets
        ), "Image with hidden objects & targets should be different than with just hiding targets"

    with sim.hide_objects(hide_robot=True):
        img_hide_objects_and_robot = sim.render()
        assert not np.allclose(
            img_hide_objects, img_hide_objects_and_robot
        ), "Hiding the robot should result in a different image"
