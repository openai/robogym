from copy import deepcopy
from glob import glob
from os.path import abspath, basename, dirname, join

import numpy as np
import pytest
from mock import patch

import robogym.utils.rotation as rotation
from robogym.envs.rearrange.blocks import make_env as make_blocks_env
from robogym.envs.rearrange.blocks_reach import make_env as make_reach_env
from robogym.envs.rearrange.common.utils import (
    geom_ids_of_body,
    load_material_args,
    recursive_dict_update,
    safe_reset_env,
)
from robogym.envs.rearrange.composer import make_env as make_composer_env
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal
from robogym.envs.rearrange.holdout import HoldoutRearrangeEnv
from robogym.mujoco.mujoco_xml import StaleMjSimError
from robogym.robot.robot_interface import (
    ControlMode,
    RobotControlParameters,
    TcpSolverMode,
)
from robogym.utils.env_utils import InvalidSimulationError, load_env
from robogym.utils.rotation import normalize_angles


def is_holdout_env(env) -> bool:
    return isinstance(env.unwrapped, HoldoutRearrangeEnv)


def is_fixed_goal_env(env) -> bool:
    # The manually designed hold-out envs which can provide the same goal across resets.
    FIXED_GOAL_ENVS = [
        "TableSetting",
        "Chessboard",
        "WordBlocks",
        "DiversityBlockRearrange",
    ]
    if any(keyword in str(env.unwrapped.__class__) for keyword in FIXED_GOAL_ENVS):
        return True

    if is_holdout_env(env):
        return len(env.unwrapped.constants.goal_args.goal_state_paths) == 1

    return False


def is_fixed_initial_state_env(env) -> bool:
    if is_holdout_env(env):
        return env.unwrapped.constants.initial_state_path is not None
    else:
        return False


def is_complex_object_env(env) -> bool:
    # For envs with complex objects, different keys should be static. Here,
    # we exclude all object-related things since the objects for YCB are not simple
    # blocks, i.e. they will move slightly when positioned due to their more complex
    # geometry.
    COMPLEX_OBJECT_ENVS = ["Ycb", "ShapeNet", "Composer"]
    return any(
        keyword in str(env.unwrapped.__class__) for keyword in COMPLEX_OBJECT_ENVS
    )


def _list_rearrange_envs(include_holdout=True, **kwargs):
    rearrange_path = abspath(join(dirname(__file__), ".."))

    # Load envs defined as python file.
    for env_path in glob(join(rearrange_path, "*.py")):
        if basename(env_path).startswith("__"):
            continue

        if basename(env_path) == "holdout.py":
            # Don't load holdout env directly.
            continue

        if "shapenet" in env_path:
            # We need to use small default mesh_scale for shapenet because objects are too big
            # and this causes collision among objects and a robot.
            shapenet_args = recursive_dict_update(
                deepcopy(kwargs),
                {"parameters": {"simulation_params": {"mesh_scale": 0.1}}},
            )
            yield load_env(env_path, **shapenet_args)
        else:
            yield load_env(env_path, **kwargs)

    # Load holdout envs defined as jsonnet.
    if include_holdout:
        for env_path in glob(join(rearrange_path, "holdouts/configs", "*.jsonnet")):
            yield load_env(f"{env_path}::make_env", **kwargs)


def test_env_basic_action():
    for env in _list_rearrange_envs():
        safe_reset_env(env)

        for _ in range(10):
            env.step(env.action_space.sample())


def test_composer_env():
    for num_max_geoms in [1, 3, 5, 8]:
        parameters = {
            "simulation_params": {"num_max_geoms": num_max_geoms, "num_objects": 3}
        }
        env = make_composer_env(parameters=parameters)

        for _ in range(10):
            safe_reset_env(env)
            env.step(env.action_space.sample())


@pytest.mark.parametrize(
    "control_mode, expected_action_dim",
    [(ControlMode.TCP_WRIST.value, 5), (ControlMode.TCP_ROLL_YAW.value, 6)],
)
def test_action_space(control_mode, expected_action_dim):
    env_args = dict(
        parameters=dict(robot_control_params=dict(control_mode=control_mode)),
    )
    for env in _list_rearrange_envs(**env_args):
        assert len(env.action_space.sample()) == expected_action_dim


def test_max_num_objects():
    """
    Test which makes sure all rearrange environments runs fine with max number of objects.
    """
    env = make_blocks_env(
        parameters={"simulation_params": {"num_objects": 8, "max_num_objects": 8}}
    )
    env.reset()
    env.step(env.action_space.sample())


REACH_THRESHOLD = 0.02
STEPS_THRESHOLD = 10


@pytest.mark.parametrize(
    "control_mode, action_dim",
    [(ControlMode.TCP_WRIST.value, 5), (ControlMode.TCP_ROLL_YAW.value, 6)],
)
def test_gripper_table_proximity(control_mode, action_dim):
    env = make_blocks_env(
        parameters={
            "n_random_initial_steps": 0,
            "robot_control_params": {"control_mode": control_mode},
        },
        starting_seed=0,
    )
    env.reset()
    # prompt arm to move in the -z direction
    action = np.zeros(action_dim)
    action[2] = -1.0
    gripper_z_obs = env.observe()["gripper_pos"][2]  # robot0:grip site offset
    z_min = np.Inf
    _, _, table_height = env.mujoco_simulation.get_table_dimensions()
    t = 0
    while gripper_z_obs > table_height + REACH_THRESHOLD and t < STEPS_THRESHOLD:
        env.unwrapped.step(action)
        gripper_z_obs = env.observe()["gripper_pos"][2]
        z_min = min(z_min, gripper_z_obs)
        t += 1
    gripper_z_obs = env.observe()["gripper_pos"][2]
    # gripper can get within REACH_THRESHOLD units to the tabletop
    assert gripper_z_obs <= table_height + REACH_THRESHOLD
    # gripper does not get closer to the table than TCP_PROTECTION_THRESHOLD
    assert z_min >= table_height


def test_randomize_camera():
    env = make_blocks_env(
        parameters={
            "simulation_params": {
                "camera_fovy_radius": 0.1,
                "camera_pos_radius": 0.007,
                "camera_quat_radius": 0.09,
            }
        }
    )

    nc = len(env.mujoco_simulation.initial_values["camera_fovy"])
    for _ in range(5):
        env.reset()
        assert np.all(
            np.abs(
                env.mujoco_simulation.mj_sim.model.cam_fovy
                - env.mujoco_simulation.initial_values["camera_fovy"]
            )
            < 0.1
        )

        assert np.allclose(
            np.linalg.norm(
                env.mujoco_simulation.mj_sim.model.cam_pos
                - env.mujoco_simulation.initial_values["camera_pos"],
                axis=1,
            ),
            0.007,
        )

        for ic in range(nc):
            # quarernion between two quat should be cos(a/2)
            angle = rotation.quat_mul(
                rotation.quat_conjugate(
                    env.mujoco_simulation.mj_sim.model.cam_quat[ic]
                ),
                env.mujoco_simulation.initial_values["camera_quat"][ic],
            )

            assert abs(angle[0] - np.cos(0.045)) < 1e-6


def test_randomize_lighting():
    env = make_blocks_env(
        parameters={
            "simulation_params": {
                "light_pos_range": 0.8,
                "light_ambient_intensity": 0.6,
                "light_diffuse_intensity": 0.4,
            }
        }
    )

    for trial in range(5):
        env.reset()
        light_pos = env.mujoco_simulation.mj_sim.model.light_pos
        light_dir = env.mujoco_simulation.mj_sim.model.light_dir

        for i in range(len(light_pos)):
            position = light_pos[i]
            direction = light_dir[i]

            pos_norm = np.linalg.norm(position)
            dir_norm = np.linalg.norm(direction)

            assert np.isclose(pos_norm, 4.0), "Lights should always be 4m from origin"
            assert np.isclose(dir_norm, 1.0), "Light direction should be unit norm"
            assert np.allclose(
                -position / pos_norm, direction
            ), "Light direction should always point to the origin"

        ambient_intensity = env.mujoco_simulation.mj_sim.model.vis.headlight.ambient
        assert np.allclose(ambient_intensity, 0.6)
        diffuse_intensity = env.mujoco_simulation.mj_sim.model.vis.headlight.diffuse
        assert np.allclose(diffuse_intensity, 0.4)


@pytest.mark.parametrize("material_name", ["painted_wood", "rubber-ball"])
def test_randomize_material(material_name):
    def str_to_np_array(s):
        return np.array([float(v) for v in s.split(" ")])

    material_args = load_material_args(material_name)
    for env in _list_rearrange_envs(
        include_holdout=False, parameters={"material_names": [material_name]}
    ):
        env.reset()
        sim = env.unwrapped.mujoco_simulation.sim

        for i in range(env.unwrapped.mujoco_simulation.num_objects):
            geom_ids = geom_ids_of_body(sim, f"object{i}")

            for geom_id in geom_ids:
                for key in ["solref", "solimp", "friction"]:
                    if key in material_args["geom"]:
                        expected = str_to_np_array(material_args["geom"][key])
                        actual = (getattr(sim.model, f"geom_{key}")[geom_id],)
                        assert np.allclose(actual, expected)


def test_invalid_goal_crash():
    class InvalidStateGoal(ObjectStateGoal):
        def next_goal(self, random_state, current_state):
            goal = super().next_goal(random_state, current_state)
            goal["goal_valid"] = False
            return goal

    env = make_blocks_env()
    env.unwrapped.goal_generation = InvalidStateGoal(env.unwrapped.mujoco_simulation)

    for fn in [env.reset, env.reset_goal, env.reset_goal_generation]:
        with pytest.raises(InvalidSimulationError):
            fn()


@pytest.mark.parametrize(
    "control_mode,tcp_solver_mode",
    [
        [ControlMode.TCP_WRIST, TcpSolverMode.MOCAP],
        [ControlMode.TCP_ROLL_YAW, TcpSolverMode.MOCAP],
        [ControlMode.TCP_ROLL_YAW, TcpSolverMode.MOCAP_IK],
        [ControlMode.TCP_WRIST, TcpSolverMode.MOCAP_IK],
    ],
)
def test_randomize_initial_robot_position(control_mode, tcp_solver_mode):
    parameters = dict(
        n_random_initial_steps=10,
        robot_control_params=dict(
            control_mode=control_mode,
            tcp_solver_mode=tcp_solver_mode,
            max_position_change=RobotControlParameters.default_max_pos_change_for_solver(
                control_mode=control_mode, tcp_solver_mode=tcp_solver_mode,
            ),
        ),
    )

    for env in _list_rearrange_envs(parameters=parameters, starting_seed=1):
        obs1 = safe_reset_env(env)
        obs2 = safe_reset_env(env)

        # robot TCP pos is randomized, gripper not in motion
        assert not np.allclose(obs2["gripper_pos"], obs1["gripper_pos"])
        assert np.allclose(obs2["gripper_velp"], 0.0, atol=3e-3)


@pytest.mark.parametrize(
    "make_env,z_action,tcp_solver_mode",
    [
        (make_blocks_env, 1, TcpSolverMode.MOCAP),
        (make_blocks_env, 1, TcpSolverMode.MOCAP_IK),
        (make_blocks_env, -1, TcpSolverMode.MOCAP),
        (make_blocks_env, -1, TcpSolverMode.MOCAP_IK),
        (make_reach_env, 1, TcpSolverMode.MOCAP),
        (make_reach_env, 1, TcpSolverMode.MOCAP_IK),
        (make_reach_env, -1, TcpSolverMode.MOCAP),
        (make_reach_env, -1, TcpSolverMode.MOCAP_IK),
    ],
)
def test_table_collision_penalty(make_env, z_action, tcp_solver_mode):
    """
    This test ensures table penalty is applied correctly when the gripper is in close proximity
    of the table, and also tests it is not spuriously applied when it is not.

    To achieve this, it applies a prescribed action in the Z direction to the arm and checks the
    gripper penalties are calculated correctly.
    :param make_env: make_env function to test
    :param z_action: action to apply in the world Z direction. All other actions will be zero.
    :param tcp_solver_mode: TCP solver mode to test.
    """
    TABLE_COLLISION_PENALTY = 0.2
    # Reward calculated by RobotEnv is of form [reward, goal_reward, success_reward].
    ENV_REWARD_IDX = 0
    SIM_REWARD_IDX = 1

    max_position_change = RobotControlParameters.default_max_pos_change_for_solver(
        control_mode=ControlMode.TCP_ROLL_YAW, tcp_solver_mode=tcp_solver_mode
    )
    env = make_env(
        parameters=dict(
            n_random_initial_steps=0,
            simulation_params=dict(
                penalty=dict(table_collision=TABLE_COLLISION_PENALTY),
            ),
            robot_control_params=dict(
                tcp_solver_mode=tcp_solver_mode,
                max_position_change=max_position_change,
            ),
        ),
        constants=dict(use_goal_distance_reward=False),
        starting_seed=0,
    ).env

    env.reset()

    # check condition at start
    expect_initial_penalty = env.unwrapped.mujoco_simulation.get_gripper_table_contact()

    if expect_initial_penalty:
        assert (
            env.get_simulation_info_reward_with_done()[SIM_REWARD_IDX]
            == TABLE_COLLISION_PENALTY
        )
    else:
        assert env.get_simulation_info_reward_with_done()[SIM_REWARD_IDX] == 0.0

    action = np.zeros_like(env.action_space.sample())
    action[2] = z_action
    for _ in range(20):
        _, reward, _, _ = env.step(action)

    expect_penalty = (
        z_action < 0
    )  # expect penalty if the action is pushing the gripper towards the table
    if expect_penalty:
        # assert env.reward() == TABLE_COLLISION_PENALTY
        assert (
            reward[ENV_REWARD_IDX] == -TABLE_COLLISION_PENALTY
        )  # goal reward should be negative
    else:
        # assert env.reward() == 0.0
        assert reward[ENV_REWARD_IDX] >= 0.0  # goal reward should be non-negative


def test_off_table_penalty():
    from robogym.envs.rearrange.blocks import make_env

    env = make_env()
    env.reset()

    _, rew, _, info = env.step(env.action_space.sample())
    assert np.equal(info.get("objects_off_table")[0], False)
    assert rew[0] >= 0.0

    with patch(
        "robogym.envs.rearrange.simulation.base.RearrangeSimulationInterface.check_objects_off_table"
    ) as mock_sim:
        mock_sim.return_value = np.array([True])
        env = make_env()
        _, rew, done, info = env.unwrapped.step(env.action_space.sample())
        assert np.equal(info.get("objects_off_table")[0], True)
        assert rew[0] == -1.0
        assert done


def test_safety_stop_penalty():
    from robogym.envs.rearrange.blocks import make_env

    SAFETY_STOP_PENALTY = 5.0
    env = make_env(
        parameters=dict(
            simulation_params=dict(penalty=dict(safety_stop=SAFETY_STOP_PENALTY),),
        ),
        constants=dict(use_goal_distance_reward=False),
        starting_seed=0,
    ).env
    env.reset()

    action = np.zeros_like(env.action_space.sample())
    action[2] = -1  # drive the arm into the table
    for _ in range(20):
        obs, reward, _, _ = env.step(action)

    assert obs["safety_stop"]
    assert reward[0] == -SAFETY_STOP_PENALTY


def test_override_state():
    from robogym.envs.rearrange.blocks import make_env

    env = make_env()
    env.reset()
    object_pos = env.mujoco_simulation.get_object_pos()
    object_rot = env.mujoco_simulation.get_object_rot()

    new_object_pos = object_pos + np.ones(3)
    new_object_rot = normalize_angles(object_rot + np.ones(3) * 0.1)

    with env.mujoco_simulation.override_object_state(new_object_pos, new_object_rot):
        assert np.array_equal(env.mujoco_simulation.get_object_pos(), new_object_pos)
        assert np.allclose(
            env.mujoco_simulation.get_object_rot(), new_object_rot, atol=1e-3
        )

    assert np.array_equal(env.mujoco_simulation.get_object_pos(), object_pos)
    assert np.allclose(env.mujoco_simulation.get_object_rot(), object_rot, atol=1e-3)


def test_teleport_to_goal():
    from robogym.envs.rearrange.blocks import make_env

    env = make_env()
    env.reset()

    for _ in range(15):
        env.step(env.action_space.sample())


def test_stale_sim_error():
    env = make_blocks_env()
    env.reset()
    sim = env.sim

    _ = sim.model
    raised = False
    env.reset()

    try:
        _ = sim.model
    except StaleMjSimError:
        raised = True

    assert raised


def test_robot_recreation_on_reset():
    env = make_blocks_env()
    initial_robot = env.robot
    env.reset()
    assert env.robot == env.mujoco_simulation.robot

    env.reset()

    assert (
        env.robot == env.mujoco_simulation.robot
    ), "Robot instance not refreshed on sim.reset"
    assert (
        initial_robot != env.robot
    ), "Expected a new robot to be created on sim reset."
