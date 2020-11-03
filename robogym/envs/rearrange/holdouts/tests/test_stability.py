import os
from typing import List

import numpy as np
from mujoco_py import const

from robogym.robot_env import RobotEnv
from robogym.utils.env_utils import load_env

ROOT_PATH = os.path.join("robogym", "envs", "rearrange", "holdouts", "configs")

# Number of steps to let the env run for while we test for stability.
NUM_STEPS = 50

MATERIALS_JSONNET_TEMPLATE = """
(import "{config_file}") + {{
    mat:: {material},
    rgba:: "{color}"
}}
"""

# Template which sets the initial state to the first goal state, used to test goal state stability.
GOAL_STATE_JSONNET_TEMPLATE = """
(import "{config_file}") + {{
    initial_state_path:: $.goal_state_paths[0],

    make_env+: {{
        args+: {{
            parameters+: {{
                # Prevent the robot from knocking over objects.
                n_random_initial_steps: 0,
            }},
        }},
    }},
}}
"""

ALL_MATERIALS = [
    # FIXME: the default material is not as stable as others, since it doesn't set solref
    # Uncomment the below once we resolve the investigation into the default material.
    # ('default.jsonnet', '0 1 0 1'),
    ("painted_wood.jsonnet", "0 0 1 1"),
    ("tangram.jsonnet", "1 0 0 1"),
    ("rubber-ball.jsonnet", "1 1 0 1"),
    ("chess.jsonnet", "1 0 1 1"),
]

# Each entry contains (config path, max speed w/ initial state, max speed w/ goal state)
# These thresholds are chosen so that we don't accidentally make things less stable; as we
# improve physics stability, we should ideally lower these thresholds.
HOLDOUT_ENVS = [
    (
        "ball_capture.jsonnet",
        0.1,
        1.0,
    ),  # Ball capture's goal state is currently unstable.
    ("chess.jsonnet", 0.2, 0.2),  # Chess is a bit more unstable.
    ("dominoes.jsonnet", 0.1, 0.1),
    ("rainbow.jsonnet", 0.1, 0.1),
    ("tangram.jsonnet", 0.1, 0.1),
    ("rainbow_build/rainbow_stack6.jsonnet", 0.1, 0.1),
    ("rainbow_build/rainbow_stack_inv6.jsonnet", 0.1, 0.1),
    ("rainbow_build/rainbow_balance.jsonnet", 0.1, 0.1),
    ("bin_packing.jsonnet", 0.1, 0.1),
    ("ball_in_mug.jsonnet", 0.1, 0.1),
    ("jenga/leaning_tower.jsonnet", 0.1, 0.1),
    ("jenga/stack6.jsonnet", 0.1, 0.1),
    ("jenga/cross.jsonnet", 0.1, 0.1),
    ("lego/easy_stack2.jsonnet", 0.1, 0.1),
    ("lego/easy_stack3.jsonnet", 0.1, 0.1),
    ("lego/stack2.jsonnet", 0.1, 0.1),
    ("lego/stack3.jsonnet", 0.1, 0.1),
    ("lego/stack5.jsonnet", 0.1, 0.1),
    ("lego/stack2L.jsonnet", 0.1, 0.1),
]

# Basic physics tests used to test all materials.
ALL_PHYSICS_TESTS = [
    "block_stacking4.jsonnet",
    "static.jsonnet",
]


def _test_static_stability(
    env: RobotEnv,
    description: str,
    max_object_speed_allowed: float = 0.5,
    max_angular_speed_allowed: float = 0.05,
    render: bool = False,
    verbose: bool = False,
):
    """
    Steps the given env for NUM_STEPS and asserts that the max object speed observed is less than
    the given threshold.
    """

    env.reset()
    sim = env.unwrapped.mujoco_simulation

    if render:
        sim.mujoco_viewer.cam.fixedcamid = 2
        sim.mujoco_viewer.cam.type = const.CAMERA_FIXED

    episode_max_speed = 0.0
    sum_max_speed = 0.0

    episode_max_ang_speed = 0.0
    sum_max_ang_speed = 0.0
    with sim.hide_target():
        for _ in range(NUM_STEPS):
            zeros = (env.action_space.nvec - 1) // 2
            env.step(zeros)

            # Compute the speed of all objects by taking the norm of the Cartesian velocity.
            speeds = [
                np.linalg.norm(sim.get_qvel(f"object{obj_idx}")[:3])
                for obj_idx in range(sim.num_objects)
            ]
            max_speed = np.max(speeds)
            sum_max_speed += max_speed
            episode_max_speed = max(max_speed, episode_max_speed)

            # Compute the angular speed of all objects (we have seen some issues with sim lead to
            # angular jittering but not translational jittering).
            ang_speeds = [
                np.linalg.norm(sim.get_qvel(f"object{obj_idx}")[3:])
                for obj_idx in range(sim.num_objects)
            ]
            max_ang_speed = np.max(ang_speeds)
            sum_max_ang_speed += max_ang_speed
            episode_max_ang_speed = max(max_ang_speed, episode_max_ang_speed)

            if render:
                env.render()

    if verbose:
        print(
            f"Max speed for {description} = {episode_max_speed:.3f}, "
            f"mean = {sum_max_speed / NUM_STEPS:.3f}"
        )
        print(
            f"Max angular speed for {description} = {episode_max_ang_speed:.3f}, "
            f"mean = {sum_max_ang_speed / NUM_STEPS:.3f}"
        )

    assert episode_max_speed <= max_object_speed_allowed, (
        f"Max speed = {episode_max_speed} > "
        f"{max_object_speed_allowed}, which is the maximum we allow. This means the following "
        f"combo is unstable: {description}"
    )
    assert episode_max_ang_speed <= max_angular_speed_allowed, (
        f"Max angular speed = "
        f"{episode_max_ang_speed} > {max_angular_speed_allowed}, which is the maximum we allow. "
        f"This means the following combo is unstable: {description}"
    )

    return episode_max_speed, episode_max_ang_speed


def _test_materials(
    test_cases: List,
    max_object_speed_allowed: float = 0.1,
    max_angular_speed: float = 0.05,
    materials: List = None,
    render: bool = False,
    verbose: bool = False,
):
    """
    Evaluates the stability of all test cases on all given materials (defaults to all available
    materials).
    """
    if not materials:
        materials = ALL_MATERIALS

    overall_max_speed = 0.0
    overall_max_ang_speed = 0.0

    for material, color in materials:
        for test in test_cases:
            path = os.path.join(ROOT_PATH, "physics_tests", "tmp_" + test)

            if material:
                parsed_material = f'(import "../../../materials/{material}")'
            else:
                parsed_material = "{}"
            jsonnet = MATERIALS_JSONNET_TEMPLATE.format(
                config_file=test, material=parsed_material, color=color
            )
            with open(path, "w") as f:
                f.write(jsonnet)

            env = load_env(path, starting_seed=0)
            os.remove(path)

            desc = f"material = {material} on test = {test}"
            env_max_speed, env_max_ang_speed = _test_static_stability(
                env,
                desc,
                max_object_speed_allowed,
                max_angular_speed,
                render=render,
                verbose=verbose,
            )

            overall_max_speed = max(overall_max_speed, env_max_speed)
            overall_max_ang_speed = max(overall_max_ang_speed, env_max_ang_speed)

    if verbose:
        print(f"\nMax object speed across all test cases = {overall_max_speed:.3f}")
        print(
            f"\nMax angular speed across all test cases = {overall_max_ang_speed:.3f}"
        )


def test_stacking():
    _test_materials(
        ["block_stacking4.jsonnet"],
        max_object_speed_allowed=0.3,
        max_angular_speed=3.0,
        materials=ALL_MATERIALS,
    )


def test_resting():
    _test_materials(
        ["static.jsonnet"],
        max_object_speed_allowed=0.1,
        max_angular_speed=1.0,
        materials=ALL_MATERIALS,
    )


def test_initial_states():
    for holdout_env, max_speed, _ in HOLDOUT_ENVS:
        path = os.path.join(ROOT_PATH, holdout_env)
        env = load_env(path, starting_seed=0)
        _test_static_stability(
            env,
            f"initial state for {holdout_env}",
            max_object_speed_allowed=max_speed,
            max_angular_speed_allowed=6.0,
        )


def test_goal_states():
    for holdout_env, _, max_speed in HOLDOUT_ENVS:
        fname = "tmp_" + holdout_env.split("/")[-1]
        path = os.path.join(ROOT_PATH, fname)

        jsonnet = GOAL_STATE_JSONNET_TEMPLATE.format(config_file=holdout_env)
        with open(path, "w") as f:
            f.write(jsonnet)

        env = load_env(path, starting_seed=0)
        os.remove(path)
        _test_static_stability(
            env,
            f"goal state for {holdout_env}",
            max_object_speed_allowed=max_speed,
            max_angular_speed_allowed=6.0,
        )
