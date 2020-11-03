import numpy as np
import pytest

from robogym.envs.rearrange.ycb import make_env


@pytest.mark.parametrize("mesh_scale", [0.5, 1.0, 1.5])
def test_mesh_centering(mesh_scale):
    # We know these meshe stls. are not center properly.
    for mesh_name in ["005_tomato_soup_can", "073-b_lego_duplo", "062_dice"]:
        env = make_env(
            parameters={
                "mesh_names": mesh_name,
                "simulation_params": {"mesh_scale": mesh_scale},
            }
        ).unwrapped
        obj_pos = env.mujoco_simulation.get_object_pos()
        bounding_pos = env.mujoco_simulation.get_object_bounding_boxes()[:, 0, :]

        assert np.allclose(obj_pos, bounding_pos, atol=5e-3)
