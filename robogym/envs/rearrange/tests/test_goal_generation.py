import mock
import numpy as np
import pytest

from robogym.envs.rearrange.blocks import make_env as make_blocks_env
from robogym.envs.rearrange.common.utils import safe_reset_env
from robogym.envs.rearrange.composer import make_env as make_composer_env
from robogym.envs.rearrange.goals.holdout_object_state import HoldoutObjectStateGoal
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal
from robogym.envs.rearrange.holdout import make_env as make_holdout_env
from robogym.utils import rotation


def test_stablize_goal_objects():
    random_state = np.random.RandomState(seed=0)

    env = make_composer_env(
        parameters={"simulation_params": {"num_objects": 3, "max_num_objects": 8}}
    ).unwrapped
    safe_reset_env(env)
    goal_gen = env.goal_generation

    # randomly sample goals
    goal_gen._randomize_goal_orientation(random_state)
    goal_positions, _ = goal_gen._sample_next_goal_positions(random_state)

    # intentionally put the target up in the air
    goal_positions[:, -1] += 0.1
    goal_gen.mujoco_simulation.set_target_pos(goal_positions)
    goal_gen.mujoco_simulation.forward()

    old_obj_pos = goal_gen.mujoco_simulation.get_object_pos()
    old_obj_rot = goal_gen.mujoco_simulation.get_object_rot()

    old_target_pos = goal_gen.mujoco_simulation.get_target_pos()
    old_target_rot = goal_gen.mujoco_simulation.get_target_rot()

    goal_gen._stablize_goal_objects()

    new_obj_pos = goal_gen.mujoco_simulation.get_object_pos()
    new_obj_rot = goal_gen.mujoco_simulation.get_object_rot()
    assert np.allclose(new_obj_pos, old_obj_pos)
    assert np.allclose(new_obj_rot, old_obj_rot)

    new_target_pos = goal_gen.mujoco_simulation.get_target_pos()
    new_target_rot = goal_gen.mujoco_simulation.get_target_rot()
    assert all(old_target_pos[:3, -1] > new_target_pos[:3, -1])
    assert not np.allclose(old_target_rot, new_target_rot)


@pytest.mark.parametrize("rot_type", ["z_axis", "block", "full"])
def test_randomize_goal_orientation(rot_type):
    random_state = np.random.RandomState(seed=0)
    env = make_blocks_env(
        parameters={"simulation_params": {"num_objects": 3, "max_num_objects": 8}},
        constants={
            "goal_args": {
                "randomize_goal_rot": True,
                "rot_randomize_type": rot_type,
                "stabilize_goal": False,  # stabilize_goal should be False to test 'full' rotation
            }
        },
    )
    safe_reset_env(env)
    goal_gen = env.goal_generation

    quats = []
    for _ in range(100):
        goal_gen._randomize_goal_orientation(random_state)
        quats.append(goal_gen.mujoco_simulation.get_target_quat(pad=False))
    quats = np.concatenate(quats, axis=0)  # [num randomization, 4]

    # there should be some variance in the randomized results
    assert quats.std() > 0.0

    if rot_type == "z_axis":
        # ensure objects are rotated along z axis only
        for i in range(quats.shape[0]):
            assert rotation.rot_z_aligned(quats[i], 0.02, include_flip=False)
    elif rot_type == "block":
        # ensure at least one face of the block is facing on top
        for i in range(quats.shape[0]):
            assert rotation.rot_xyz_aligned(quats[i], 0.02)
    elif rot_type == "full":
        # ensure that at least one randomize object has weird pose that any of the face does not
        # face the top direction
        rot_xyz_aligned = []
        for i in range(quats.shape[0]):
            rot_xyz_aligned.append(rotation.rot_xyz_aligned(quats[i], 0.02))
        assert all(rot_xyz_aligned) is False


@mock.patch(
    "robogym.envs.rearrange.common.base.sample_group_counts",
    mock.MagicMock(return_value=[3, 2, 1]),
)
def test_relative_pos_for_duplicated_objects():
    current_goal_state = {
        "obj_rot": np.array(
            [
                [0, 0, np.pi / 2],
                [np.pi / 2, 0, 0],
                [0, np.pi / 2, 0],
                [np.pi / 2, 0, 0],
                [0, np.pi / 2, 0],
                [np.pi / 2, 0, 0],
            ]
        ),
        "obj_pos": np.array(
            [[2, 2, 2], [3, 3, 4], [0, 1, 1], [1, 2, 3], [1, 1, 1], [5, 5, 6]]
        ),
    }
    target_goal_state = {
        "obj_rot": np.array(
            [
                [0, np.pi / 2, 0],
                [0, 0, np.pi / 2],
                [np.pi / 2, 0, 0],
                [0, np.pi / 2, 0],
                [np.pi / 2, 0, 0],
                [np.pi / 2, 0, 0],
            ]
        ),
        "obj_pos": np.array(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1], [1, 2, 3], [6, 5, 6]]
        ),
    }

    env = make_composer_env(
        parameters={
            "simulation_params": {
                "num_objects": 6,
                "max_num_objects": 6,
                "num_max_geoms": 1,
            }
        }
    )
    safe_reset_env(env)

    goal_generator = ObjectStateGoal(env.unwrapped.mujoco_simulation)
    relative_goal = goal_generator.relative_goal(target_goal_state, current_goal_state)

    assert np.allclose(
        relative_goal["obj_pos"],
        np.array([[0, 0, 0], [0, 0, -1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    )
    assert np.allclose(relative_goal["obj_rot"], np.zeros((6, 3)))


def test_relative_distance():
    pos = np.array(
        [
            [1.35620895, 0.5341387, 0.48623528],
            [1.37134474, 0.53952575, 0.526087],
            [1.3713598, 0.53945007, 0.5056565],
        ]
    )
    goal_state = {
        "obj_pos": pos,
        "obj_rot": np.zeros_like(pos),
    }

    current_pos = pos.copy()
    rnd_pos = np.array([np.random.random(), np.random.random(), 0])
    current_pos += rnd_pos

    current_state = {
        "obj_pos": current_pos,
        "obj_rot": np.zeros_like(current_pos),
    }

    env = make_holdout_env()
    safe_reset_env(env)
    goal = HoldoutObjectStateGoal(env.unwrapped.mujoco_simulation)
    dist = goal.goal_distance(goal_state, current_state)

    assert np.allclose(dist["rel_dist"], np.zeros(3))
