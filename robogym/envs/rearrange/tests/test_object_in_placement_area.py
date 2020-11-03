import numpy as np
import pytest

from robogym.envs.rearrange.blocks import make_env

KEYS_TO_MASK = [
    "goal_obj_pos",
    "goal_obj_rot",
    "rel_goal_obj_pos",
    "rel_goal_obj_rot",
    "obj_pos",
    "obj_rot",
    "obj_rel_pos",
    "obj_vel_pos",
    "obj_vel_rot",
    "obj_gripper_contact",
    "obj_bbox_size",
    "obj_colors",
]


@pytest.mark.parametrize(
    "obj_pos,in_placement_area,margin",
    [
        ([[1.45, 0.68, 0.5]], [True], 0.02),  # Center of placement area.
        ([[1.15, 0.39, 0.5]], [True], 0.02),  # top left in boundary
        ([[1.10, 0.39, 0.5]], [False], 0.02),  # top left out of boundary
        (
            [[1.10, 0.39, 0.5]],
            [True],
            0.1,
        ),  # top left close to boundary with a big margin
        ([[1.75, 0.97, 0.5]], [True], 0.02),  # bottom right in boundary
        ([[1.80, 1.0, 0.5]], [False], 0.02),  # bottom right out of boundary
        ([[1.15, 0.97, 0.5]], [True], 0.02),  # top right in boundary
        ([[1.10, 1.0, 0.5]], [False], 0.02),  # top right out of boundary
        ([[1.75, 0.39, 0.5]], [True], 0.02),  # bottom left in boundary
        ([[1.75, 0.36, 0.5]], [False], 0.02),  # bottom left out of boundary
        (
            [[1.75, 0.36, 0.5]],
            [True],
            0.1,
        ),  # bottom close to boundary with a big margin
        # Some combinations
        ([[1.15, 0.39, 0.5], [1.10, 0.39, 0.5]], [True, False], 0.02),
        ([[1.80, 1.0, 0.5], [1.15, 0.97, 0.5]], [False, True], 0.02),
        (
            [[1.80, 1.0, 0.5], [1.10, 1.0, 0.5], [1.75, 0.39, 0.5]],
            [False, False, True],
            0.02,
        ),
    ],
)
def test_single_obj_in_placement_area(obj_pos, in_placement_area, margin):
    in_placement_area = np.array(in_placement_area)
    n_obj = len(obj_pos)
    max_obj = 12
    env = make_env(
        parameters={
            "simulation_params": {"num_objects": n_obj, "max_num_objects": max_obj}
        },
    )
    env.reset()
    sim = env.unwrapped.mujoco_simulation
    assert np.array_equal(
        in_placement_area,
        sim.check_objects_in_placement_area(np.array(obj_pos), margin=margin),
    )

    obj_pos_with_padding = np.array(obj_pos + list(np.zeros((max_obj - n_obj, 3))))
    assert obj_pos_with_padding.shape == (max_obj, 3)

    with_padding = sim.check_objects_in_placement_area(
        obj_pos_with_padding, margin=margin
    )
    assert len(with_padding) == max_obj
    assert np.array_equal(in_placement_area, with_padding[:n_obj])
    assert np.all(in_placement_area[n_obj:])

    no_padding = sim.check_objects_in_placement_area(np.array(obj_pos), margin=margin)
    assert len(no_padding) == len(obj_pos)
    assert np.array_equal(in_placement_area, no_padding)


@pytest.mark.parametrize("should_mask", [True, False])
@pytest.mark.parametrize(
    "obj_pos,in_placement_area",
    [
        ([[1.45, 0.68, 0.5]], [True]),
        ([[1.15, 0.39, 0.5], [1.10, 0.39, 0.5]], [True, False]),
        ([[1.80, 1.0, 0.5], [1.15, 0.97, 0.5]], [False, True]),
        ([[1.80, 1.0, 0.5], [1.10, 1.0, 0.5], [1.75, 0.39, 0.5]], [False, False, True]),
    ],
)
def test_mask_observation(obj_pos, in_placement_area, should_mask):
    n_obj = len(obj_pos)
    obj_pos = np.array(obj_pos)
    in_placement_area_padded = np.array(in_placement_area + [True] * (3 - n_obj))
    expected_mask = in_placement_area_padded.astype(np.float).reshape(-1, 1)

    env = make_env(
        parameters={"simulation_params": {"num_objects": n_obj, "max_num_objects": 3}},
        constants={"mask_obs_outside_placement_area": should_mask},
    )
    env.reset()
    env.unwrapped.mujoco_simulation.set_object_pos(np.array(obj_pos))
    env.unwrapped.mujoco_simulation.forward()
    env.unwrapped._goal["goal_objects_in_placement_area"] = in_placement_area_padded
    obs = env.observe()

    sim = env.unwrapped.mujoco_simulation
    assert in_placement_area == list(sim.check_objects_in_placement_area(obj_pos))

    for k in KEYS_TO_MASK:
        masked_k = f"masked_{k}"
        if not should_mask:
            assert masked_k not in obs
        else:
            assert np.array_equal(obs["placement_mask"], expected_mask)
            assert np.array_equal(obs["goal_placement_mask"], expected_mask)
            for i in range(n_obj):
                if in_placement_area[i]:
                    assert np.all(obs[masked_k][i] == obs[k][i])
                else:
                    # if outside the placement area, mask it.
                    assert np.all(obs[masked_k][i] == np.zeros_like(obs[k][i]))


@pytest.mark.parametrize(
    "obj_pos,in_placement_area",
    [
        (
            [
                [1.45, 0.68, 0.5],  # in the middle of the placement area
                [1.45, 0.395, 0.5],  # on the left edge
                [1.45, 0.34, 0.5],  # within the margin
                [1.45, 0.25, 0.5],
            ],  # outside the margin
            [True, True, None, False],
        )
    ],
)
def test_soft_mask_observation(obj_pos, in_placement_area):
    env = make_env(parameters={"simulation_params": {"num_objects": len(obj_pos)}})
    env.reset()
    sim = env.unwrapped.mujoco_simulation
    stochastic_mask = set()
    for _ in range(20):
        mask = sim.check_objects_in_placement_area(
            np.array(obj_pos), soft=True, margin=0.1
        )
        for i in range(len(in_placement_area)):
            if in_placement_area[i] is None:
                stochastic_mask.add(mask[i])
            else:
                assert in_placement_area[i] == mask[i]

    assert len(stochastic_mask) == 2
