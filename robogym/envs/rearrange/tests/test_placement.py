import numpy as np
import pytest

from robogym.envs.rearrange.common.utils import PlacementArea


@pytest.mark.parametrize(
    "used_table_portion,expected_placement_area",
    [
        (
            1.0,
            PlacementArea(
                offset=(0.3038, 0.38275, 0.06648), size=(0.6075, 0.58178, 0.26)
            ),
        ),
        (
            0.8,
            PlacementArea(
                offset=(0.3645, 0.44093, 0.06648), size=(0.486, 0.46542, 0.26)
            ),
        ),
        (
            0.6,
            PlacementArea(
                offset=(0.4253, 0.49911, 0.06648), size=(0.3645, 0.3491, 0.26)
            ),
        ),
        (
            0.4,
            PlacementArea(
                offset=(0.486, 0.55728, 0.06648), size=(0.243, 0.23271, 0.26)
            ),
        ),
    ],
)
def test_block_placement(used_table_portion, expected_placement_area):
    from robogym.envs.rearrange.blocks import make_env

    parameters = dict(simulation_params=dict(used_table_portion=used_table_portion))

    env = make_env(parameters=parameters)
    env.reset()

    placement_area = env.mujoco_simulation.get_placement_area()
    assert np.allclose(placement_area.size, expected_placement_area.size, atol=1e-4)
    assert np.allclose(placement_area.offset, expected_placement_area.offset, atol=1e-4)


@pytest.mark.parametrize(
    "num_objects,object_scale_high,object_scale_low,normalize_mesh",
    [
        (1, 0.0, 0.0, True),
        (1, 0.0, 0.0, False),
        (8, 0.0, 0.0, True),
        (8, 0.0, 0.0, False),
        (16, 0.0, 0.0, True),
        (32, 0.0, 0.0, True),
        (32, 0.7, 0.0, True),
        (32, 0.0, 0.5, True),
    ],
)
def test_ycb_placement(
    num_objects, object_scale_high, object_scale_low, normalize_mesh
):
    # Tests initial/goal object placement in a variety of scenarios.
    from robogym.envs.rearrange.ycb import make_env

    parameters = dict(
        simulation_params=dict(
            num_objects=num_objects,
            max_num_objects=num_objects,
            mesh_scale=1.0 if normalize_mesh else 0.6,
        ),
        object_scale_high=object_scale_high,
        object_scale_low=object_scale_low,
    )
    constants = dict(normalize_mesh=normalize_mesh,)
    env = make_env(parameters=parameters, constants=constants)

    for i in range(5):
        # This will throw an exception if placement is invalid.
        env.reset()
