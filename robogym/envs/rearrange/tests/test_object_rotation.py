import os
from typing import Tuple

import numpy as np
import pytest
from mock import patch
from numpy.random import RandomState
from numpy.testing import assert_allclose

from robogym.envs.rearrange.blocks import make_env as make_blocks_env
from robogym.envs.rearrange.common.mesh import MeshRearrangeEnv
from robogym.envs.rearrange.common.utils import rotate_bounding_box, safe_reset_env
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal
from robogym.envs.rearrange.ycb import make_env as make_ycb_env
from robogym.mujoco.mujoco_xml import ASSETS_DIR
from robogym.utils import rotation


class SimpleGeomMeshRearrangeEnv(MeshRearrangeEnv):
    def _sample_object_meshes(self, num_groups: int):
        geom = self.parameters.mesh_names[0]
        return [[os.path.join(ASSETS_DIR, "stls/geom", f"{geom}.stl")]] * num_groups


class FixedRotationGoal(ObjectStateGoal):
    def __init__(self, *args, fixed_z=0.0, **kwargs):
        self.fixed_z = fixed_z
        super().__init__(*args, **kwargs)

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        object_pos = self.mujoco_simulation.get_object_pos()[
            : self.mujoco_simulation.num_objects
        ]
        return object_pos.copy(), True

    def _sample_next_goal_orientations(self, random_state: RandomState) -> np.ndarray:
        num_objects = self.mujoco_simulation.num_objects
        z_quat = rotation.quat_from_angle_and_axis(
            angle=np.array([self.fixed_z] * num_objects),
            axis=np.array([[0, 0, 1.0]] * num_objects),
        )
        return rotation.quat_mul(
            z_quat, self.mujoco_simulation.get_target_quat(pad=False)
        )


def _run_rotation_test(make_env, parameters, constants, angles_to_dists, atol=0.005):
    env = make_env(parameters=parameters, constants=constants)
    env.unwrapped.goal_generation = FixedRotationGoal(
        env.mujoco_simulation, args=constants["goal_args"]
    )

    for angle, dist in angles_to_dists.items():
        # Attempt 10 times, since sometimes the object is placed so that the gripper bumps
        # into it.
        for attempt in range(10):
            env.unwrapped.goal_generation.fixed_z = angle
            safe_reset_env(env)
            info = env.goal_info()[-1]
            assert "goal_dist" in info
            if np.allclose(info["goal_dist"]["obj_rot"], dist, atol=atol):
                break
        assert_allclose(info["goal_dist"]["obj_rot"], dist, atol=atol)


def test_mod90_rotation_blocks():
    parameters = {}
    constants = {
        "goal_args": {"rot_dist_type": "mod90", "randomize_goal_rot": True},
        "success_threshold": {"obj_pos": 0.05, "obj_rot": 0.2},
    }

    angles_to_dists = {
        # Basic cases.
        0.0: 0.0,
        0.05: 0.05,
        0.1: 0.1,
        # Rotating in different direction has same distance.
        -0.05: 0.05,
        -0.1: 0.1,
        # mod90-specific cases
        np.pi: 0.0,
        -np.pi: 0.0,
        2 * np.pi: 0.0,
        np.pi / 2: 0.0,
        -np.pi / 2: 0.0,
        np.pi / 2 + 0.05: 0.05,
        np.pi / 2 - 0.05: 0.05,
        np.pi / 4: np.pi / 4,
        np.pi / 4 + 0.05: np.pi / 4 - 0.05,
        np.pi / 4 - 0.05: np.pi / 4 - 0.05,
    }

    _run_rotation_test(make_blocks_env, parameters, constants, angles_to_dists)


def test_mod180_rotation_blocks():
    parameters = {}
    constants = {
        "goal_args": {"rot_dist_type": "mod180", "randomize_goal_rot": True},
        "success_threshold": {"obj_pos": 0.05, "obj_rot": 0.2},
    }

    angles_to_dists = {
        # Basic cases.
        0.0: 0.0,
        0.05: 0.05,
        0.1: 0.1,
        # Rotating in different direction has same distance.
        -0.05: 0.05,
        -0.1: 0.1,
        # mod180-specific cases
        np.pi: 0.0,
        -np.pi: 0.0,
        2 * np.pi: 0.0,
        np.pi / 2: np.pi / 2,
        -np.pi / 2: np.pi / 2,
        np.pi / 4: np.pi / 4,
        -np.pi / 4: np.pi / 4,
        np.pi * 3 / 4: np.pi / 4,
        -np.pi * 3 / 4: np.pi / 4,
        np.pi * 5 / 4: np.pi / 4,
        -np.pi * 5 / 4: np.pi / 4,
        np.pi + 0.05: 0.05,
        np.pi - 0.05: 0.05,
        np.pi / 2 + 0.05: np.pi / 2 - 0.05,
        np.pi / 2 - 0.05: np.pi / 2 - 0.05,
    }

    _run_rotation_test(make_blocks_env, parameters, constants, angles_to_dists)


def test_full_rotation_blocks():
    parameters = {}
    constants = {
        "goal_args": {"rot_dist_type": "full", "randomize_goal_rot": True},
        "success_threshold": {"obj_pos": 0.05, "obj_rot": 0.2},
    }

    angles_to_dists = {
        # Basic cases.
        0.0: 0.0,
        0.05: 0.05,
        0.1: 0.1,
        # Rotating in different direction has same distance.
        -0.05: 0.05,
        -0.1: 0.1,
        # full-specific cases
        np.pi: np.pi,
        np.pi / 2: np.pi / 2,
        -np.pi / 2: np.pi / 2,
        np.pi + 0.05: np.pi - 0.05,
        np.pi - 0.05: np.pi - 0.05,
        2 * np.pi: 0.0,
    }

    _run_rotation_test(make_blocks_env, parameters, constants, angles_to_dists)


def test_bounding_box_rotation():
    # Identity quaternion should have no effect.
    bounding_box = (np.zeros(3), np.ones(3))
    quat = np.array([1, 0, 0, 0])
    rotated_bounding_box = rotate_bounding_box(bounding_box, quat)
    assert_allclose(bounding_box, rotated_bounding_box)

    # 90 degree rotations should have no effect.
    bounding_box = (np.zeros(3), np.ones(3))
    for parallel in rotation.get_parallel_rotations():
        quat = rotation.euler2quat(parallel)
        rotated_bounding_box = rotate_bounding_box(bounding_box, quat)
        assert_allclose(bounding_box, rotated_bounding_box)

    # 45 degree rotation around parallel axis.
    for axis in [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]:
        bounding_box = (np.zeros(3), np.ones(3))
        quat = rotation.quat_from_angle_and_axis(
            np.array(np.deg2rad(45)), axis=np.array(axis)
        )
        assert quat.shape == (4,)
        rotated_bounding_box = rotate_bounding_box(bounding_box, quat)
        assert_allclose(rotated_bounding_box[0], np.zeros(3))

        axis_idx = np.argmax(axis)
        ref_size = np.ones(3) * np.sqrt(2)
        ref_size[axis_idx] = 1.0
        assert_allclose(rotated_bounding_box[1], ref_size)


def test_full_rotation_ycb():
    parameters = {"mesh_names": ["029_plate"]}
    constants = {
        "goal_args": {"rot_dist_type": "full", "randomize_goal_rot": True},
        "success_threshold": {"obj_pos": 0.05, "obj_rot": 0.2},
    }

    angles_to_dists = {
        # Basic cases.
        0.0: 0.0,
        0.05: 0.05,
        0.1: 0.1,
        2 * np.pi: 0.0,
        # Rotating in different direction has same distance.
        -0.05: 0.05,
        -0.1: 0.1,
        # full-specific cases
        np.pi: np.pi,
        np.pi / 2: np.pi / 2,
        -np.pi / 2: np.pi / 2,
        np.pi + 0.05: np.pi - 0.05,
        np.pi - 0.05: np.pi - 0.05,
        2 * np.pi: 0.0,
    }

    _run_rotation_test(make_ycb_env, parameters, constants, angles_to_dists)


# FIXME: The test is flaky when use_bbox_precheck = True.
@pytest.mark.parametrize("use_bbox_precheck", [False])
@pytest.mark.parametrize(
    "mesh_names,z_rots,matches",
    [
        # rotation symmetry around z axis.
        (
            [
                "005_tomato_soup_can",
                "029_plate",
                "024_bowl",
                "013_apple",
                "017_orange",
                "007_tuna_fish_can",
            ],
            [0.0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
            [True, True, True, True],
        ),
        # mod90 symmetry around z axis.
        (
            ["062_dice", "070-b_colored_wood_blocks"],
            [0.0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
            [True, False, True, False],
        ),
        # mod180 symmetry around z axis.
        (
            [
                "061_foam_brick",
                "009_gelatin_box",
                "008_pudding_box",
                "003_cracker_box",
                "004_sugar_box",
            ],
            [0.0, np.pi / 4, np.pi / 2, np.pi],
            [True, False, False, True],
        ),
        # asymmetric objects.
        (
            [
                "037_scissors",
                "011_banana",
                "030_fork",
                "050_medium_clamp",
                "048_hammer",
                "072-b_toy_airplane",
                "035_power_drill",
                "033_spatula",
                "042_adjustable_wrench",
            ],
            [0.0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi],
            [True, False, False, False, False, False],
        ),
    ],
)
def test_icp_rotation_goal_mesh(use_bbox_precheck, mesh_names, z_rots, matches):
    for mesh_name in mesh_names:
        env = make_ycb_env(
            constants={
                "goal_args": {
                    "randomize_goal_rot": True,
                    "rot_dist_type": "icp",
                    "icp_use_bbox_precheck": use_bbox_precheck,
                }
            },
            parameters={"n_random_initial_steps": 0, "mesh_names": [mesh_name]},
        ).unwrapped

        _run_icp_rotation_goal_test(env, z_rots, matches)


@pytest.mark.parametrize(
    "mesh_names,init_rot,z_rots,matches",
    [
        # mod180 symmetry around z axis.
        (
            ["capsule"],
            np.array([0.0, np.pi / 2, 0.0]),
            [0.0, np.pi / 4, np.pi / 2, np.pi],
            [True, False, False, True],
        ),
        # rotation symmetry around z axis.
        (
            ["cylinder", "capsule", "halfsphere", "sphere320", "sphere1280"],
            np.zeros(3),
            [0.0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
            [True, True, True, True],
        ),
        # mod90 symmetry around z axis.
        (
            ["cube"],
            np.zeros(3),
            [0.0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
            [True, False, True, False],
        ),
    ],
)
def test_icp_rotation_goal_geom(mesh_names, init_rot, z_rots, matches):
    for mesh_name in mesh_names:
        env = SimpleGeomMeshRearrangeEnv.build(
            constants={
                "goal_args": {"randomize_goal_rot": True, "rot_dist_type": "icp"}
            },
            parameters={"n_random_initial_steps": 0, "mesh_names": [mesh_name]},
        ).unwrapped

        _run_icp_rotation_goal_test(env, z_rots, matches, init_rot=init_rot)


def _run_icp_rotation_goal_test(env, z_rots, matches, init_rot=np.zeros(3)):
    init_quat = rotation.euler2quat(init_rot)

    for z_rot, match in zip(z_rots, matches):

        def mock_randomize_goal_rot(*args, **kwargs):
            z_quat = rotation.quat_from_angle_and_axis(z_rot, np.array([0.0, 0.0, 1]))
            target_quat = rotation.quat_mul(z_quat, init_quat)
            env.mujoco_simulation.set_object_quat(np.array([init_quat]))
            env.mujoco_simulation.set_target_quat(np.array([target_quat]))
            env.mujoco_simulation.forward()

        with patch(
            "robogym.envs.rearrange.goals.object_state.ObjectStateGoal."
            "_randomize_goal_orientation"
        ) as mock_obj:
            mock_obj.side_effect = mock_randomize_goal_rot

            safe_reset_env(env)

            # set the object position same as the target position.
            sim = env.unwrapped.mujoco_simulation
            sim.set_object_pos(sim.get_target_pos())
            sim.forward()

            goal_dist = env.unwrapped.goal_generation.goal_distance(
                env.unwrapped._goal, env.unwrapped.goal_generation.current_state()
            )
            rot_dist = goal_dist["obj_rot"]

            if match:
                assert np.all(rot_dist < 0.2)
            else:
                assert np.all(rot_dist > 0.2)
