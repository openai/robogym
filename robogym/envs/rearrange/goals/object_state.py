from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import attr
import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import (
    place_objects_in_grid,
    place_objects_with_no_constraint,
    stabilize_objects,
    update_object_z_coordinate,
)
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.goal.goal_generator import GoalGenerator
from robogym.utils import rotation
from robogym.utils.icp import ICP

PARALLEL_QUATS = [rotation.euler2quat(x) for x in rotation.get_parallel_rotations()]
PARALLEL_QUATS_180 = [
    rotation.euler2quat(x) for x in rotation.get_parallel_rotations_180()
]


def euler_angle_difference_single_pair(
    q1: np.ndarray, q2: np.ndarray, parallel_quats: List[np.ndarray]
) -> np.ndarray:
    assert q1.shape == q2.shape == (4,)

    if np.allclose(q1, q2):
        return np.zeros(3)

    diffs = np.array(
        [
            rotation.quat_difference(rotation.quat_mul(q1, parallel_quat), q2)
            for parallel_quat in parallel_quats
        ]
    )
    dists = rotation.quat_magnitude(diffs)
    indices = np.argmin(dists, axis=0)
    return rotation.quat2euler(diffs[indices])


def euler_angle_difference(
    goal_state: dict, current_state: dict, mode: str
) -> np.ndarray:
    """
    This method applies
    - all 24 possible mod 90 degree rotations to given euler angle (if mode = 'mod90');
    - or, all 4 possible mod 180 degree rotations to given euler angle (if mode = 'mod180').
    and return the one with minimum quat distance.
    """
    assert mode in ("mod90", "mod180")
    parallel_quats = PARALLEL_QUATS if mode == "mod90" else PARALLEL_QUATS_180

    q1 = rotation.euler2quat(goal_state["obj_rot"])
    q2 = rotation.euler2quat(current_state["obj_rot"])

    return np.array(
        [
            euler_angle_difference_single_pair(q1[i], q2[i], parallel_quats)
            for i in range(q1.shape[0])
        ]
    )


def full_euler_angle_difference(goal_state: dict, current_state: dict):
    return rotation.subtract_euler(goal_state["obj_rot"], current_state["obj_rot"])


def _random_quat_along_z(num_objects: int, random_state: RandomState):
    """ Internal helper function generating random rotation quat along z axis """
    quat = rotation.quat_from_angle_and_axis(
        angle=random_state.uniform(low=0.0, high=2.0 * np.pi, size=num_objects),
        axis=np.array([[0, 0, 1.0]] * num_objects),
    )
    return quat


def randomize_quaternion_along_z(
    mujoco_simulation: RearrangeSimulationInterface, random_state: RandomState
):
    """ Rotate goal along z axis and return the rotated quat of the goal """
    quat = _random_quat_along_z(mujoco_simulation.num_objects, random_state)
    return rotation.quat_mul(quat, mujoco_simulation.get_target_quat(pad=False))


def randomize_quaternion_block(
    mujoco_simulation: RearrangeSimulationInterface, random_state: RandomState
):
    """
    Rotate goal and return the rotated quat of the goal. Assuming objects are blocks,
    this function randomly choose any face of the block as top face, and rotate it along z axis
    """
    num_objects = mujoco_simulation.num_objects

    z_quat = _random_quat_along_z(num_objects, random_state)
    face_quat_indices = random_state.randint(
        low=0, high=len(PARALLEL_QUATS), size=num_objects
    )
    face_quat = np.array([PARALLEL_QUATS[i] for i in face_quat_indices])
    target_quat = mujoco_simulation.get_target_quat(pad=False)
    return rotation.quat_mul(z_quat, rotation.quat_mul(target_quat, face_quat))


def randomize_quaternion_full(
    mujoco_simulation: RearrangeSimulationInterface, random_state: RandomState
):
    """
    Rotate goal in any orientation and return the rotated quat of the goal. The goal may be in
    an unstable position depending on the shape of the objects.
    """
    quat = np.array(
        [
            rotation.uniform_quat(random_state)
            for _ in range(mujoco_simulation.num_objects)
        ]
    )
    return rotation.quat_mul(quat, mujoco_simulation.get_target_quat(pad=False))


@attr.s(auto_attribs=True)
class GoalArgs:
    # If true randomize goal orientation.
    randomize_goal_rot: bool = False

    # Type of rotation distance calculation for goal.
    rot_dist_type: str = attr.ib(
        default="full",
        validator=attr.validators.in_(["full", "mod90", "mod180", "icp"]),
    )

    # Type of rotation randomization for goal.
    # z_axis: randomize rotation along z axis
    # block: assume the object is box. first randomize face direction and then rotate along z axis
    # full: fully rotate object in any direction
    rot_randomize_type: str = attr.ib(
        default="z_axis", validator=attr.validators.in_(["z_axis", "block", "full"])
    )

    # Max number of vertices to sample from object body for icp calculation.
    icp_max_num_vertices: int = 500

    # Threshold for nearest neighbor error for icp. The absolute threshold value
    # is defined as bounding_box_size * error_threshold
    icp_error_threshold: float = 0.15

    # If true, check the bounding box IoU before calling ICP rot distance measure.
    icp_use_bbox_precheck: bool = False

    # If true, step simulation to stabilize goal objects.
    stabilize_goal: bool = False

    # whether to render robot in the goal image
    p_goal_hide_robot: float = 1.0

    # If true, when calling check_objects_in_placement_area(), we will label an object
    # within the margin area stochastically.
    soft_mask: bool = False
    # The margin of error we consider when calling check_objects_in_placement_area().
    mask_margin: float = 0.02

    @property
    def goal_hide_robot(self) -> bool:
        return np.random.rand() < self.p_goal_hide_robot

    # Parameters for adding pick-and-place and stacking goals into blocks training env.
    height_range: Tuple[float, float] = (0.05, 0.25)
    pickup_proba: float = 0.0
    stacking_proba: float = 0.0


class ObjectStateGoal(GoalGenerator):
    """
    This is the default goal state generator for the rearrangement task. The goal can be
    places everywhere on the table without restrictions.
    """

    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs] = GoalArgs(),
    ):
        """
        :param mujoco_simulation: The mujoco simulation instance.
        :param args: All goal related arguments.
        """
        self.mujoco_simulation = mujoco_simulation

        if isinstance(args, GoalArgs):
            self.args = args
        else:
            assert isinstance(args, dict)
            self.args = GoalArgs(**args)

        rot_dist_funcs: Dict[str, Callable[[dict, dict], np.ndarray]] = {
            "full": full_euler_angle_difference,
            "mod90": lambda g1, g2: euler_angle_difference(g1, g2, "mod90"),
            "mod180": lambda g1, g2: euler_angle_difference(g1, g2, "mod180"),
            "icp": self._icp_euler_angle_difference,
        }

        self.rot_dist_func = rot_dist_funcs[self.args.rot_dist_type]

        rot_randomize_funcs: Dict[
            str, Callable[[RearrangeSimulationInterface, RandomState], np.ndarray]
        ] = {
            "z_axis": randomize_quaternion_along_z,
            "block": randomize_quaternion_block,
            "full": randomize_quaternion_full,
        }
        self.rot_randomize_func = rot_randomize_funcs[self.args.rot_randomize_type]

        # Only used for vertices sampling
        self._vertices_random_state = np.random.RandomState(seed=0)

        super().__init__()

    def _bounding_box_ious(self, goal_state: Dict, current_state: Dict) -> List[float]:
        """
        For each object, check the object bounding boxes of the object and the target,
        and then compute the overlap (IoU = interaction over union) between two top-down
        views of these two boxes.

        :returns: a list of object-target bounding box IoU metrics, one per object.
        """
        assert (
            "bounding_box" in current_state
        ), "Current object bounding box is not available in goal state"
        assert (
            "bounding_box" in goal_state
        ), "Target object bounding box is not available in goal state"

        bbox_ious = []

        for i in range(self.mujoco_simulation.num_objects):
            opos, (ox, oy, oz) = current_state["bounding_box"][i]
            tpos, (tx, ty, tz) = goal_state["bounding_box"][i]

            # (x1, y1) is top left corner and (x2, y2) is the bottom right corner.
            x1 = max(opos[0] - ox, tpos[0] - tx)
            y1 = min(opos[1] + oy, tpos[1] + ty)
            x2 = min(opos[0] + ox, tpos[0] + tx)
            y2 = max(opos[1] - oy, tpos[1] - ty)

            inter_area = abs(max((x2 - x1, 0)) * max((y1 - y2), 0))
            if inter_area == 0:
                iou = 0.0
            else:
                oarea = ox * oy * 4
                tarea = tx * ty * 4
                iou = inter_area / (oarea + tarea - inter_area)

            bbox_ious.append(iou)

        return bbox_ious

    def _icp_euler_angle_difference(self, goal_state: dict, current_state: dict):
        assert (
            "vertices" in current_state
        ), "vertices not available in current goal state"
        assert "icp" in goal_state, "ICP not available in target goal state"
        angles = np.zeros_like(current_state["obj_rot"])

        if self.args.icp_use_bbox_precheck:
            bbox_ious = self._bounding_box_ious(goal_state, current_state)
        else:
            bbox_ious = np.ones(self.mujoco_simulation.num_objects)

        for i, (ov, icp, bbox_iou) in enumerate(
            zip(current_state["vertices"], goal_state["icp"], bbox_ious,)
        ):
            # Use an arbitrarily large rotation distance as a default value.
            angles[i] = [np.pi / 2, np.pi / 2, np.pi / 2]
            if bbox_iou > 0.75:
                # Use ICP to compute the rotation distance only when the bounding box IoU is
                # large enough. Note that this check only makes sense for sparse reward.
                # In the foreseen future, it is very unlikely we will use dense reward.

                # Randomly sample vertices from object.
                ov_indices = self._vertices_random_state.permutation(ov.shape[0])[
                    : self.args.icp_max_num_vertices
                ]

                mat = icp.compute(ov[ov_indices])

                if mat is not None:
                    angles[i] = rotation.mat2euler(mat)

        return angles

    def _stablize_goal_objects(self):
        # read current object position and rotation.
        object_pos = self.mujoco_simulation.get_object_pos(pad=False).copy()
        object_quat = self.mujoco_simulation.get_object_quat(pad=False).copy()

        # set object pos/rot as target pos/rot
        self.mujoco_simulation.set_object_pos(
            self.mujoco_simulation.get_target_pos(pad=False)
        )
        self.mujoco_simulation.set_object_quat(
            self.mujoco_simulation.get_target_quat(pad=False)
        )
        self.mujoco_simulation.forward()

        # Recompute z position of objects based on new bounding box computed from new orientation
        updated = update_object_z_coordinate(
            self.mujoco_simulation.get_object_pos(pad=False),
            # Do not use target_bounding_boxes here because we stabilize target by copying its
            # state to object before stabilizing it.
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
        )
        self.mujoco_simulation.set_object_pos(updated)
        self.mujoco_simulation.forward()

        stabilize_objects(self.mujoco_simulation)

        # read and set stabilized goal pos/rot
        self.mujoco_simulation.set_target_pos(
            self.mujoco_simulation.get_object_pos(pad=False)
        )
        self.mujoco_simulation.set_target_rot(
            self.mujoco_simulation.get_object_rot(pad=False)
        )
        self.mujoco_simulation.forward()

        # restore object pos/rot back to original
        self.mujoco_simulation.set_object_pos(object_pos)
        self.mujoco_simulation.set_object_quat(object_quat)
        self.mujoco_simulation.forward()

    def _update_simulation_for_next_goal(
        self, random_state: RandomState
    ) -> Tuple[bool, Dict[str, np.ndarray]]:
        """
        Sample new goal configs and returns a tuple of:
            - a bool flag whether the goal is valid.
            - the new goal config.
        """
        # Sample target object orientations.
        if self.args.randomize_goal_rot:
            self._randomize_goal_orientation(random_state)

        # Sample target object positions.
        goal_positions, goal_valid = self._sample_next_goal_positions(random_state)
        if goal_valid:
            self.mujoco_simulation.set_target_pos(goal_positions)
            self.mujoco_simulation.forward()

        if self.args.stabilize_goal:
            self._stablize_goal_objects()

        goal = {
            "obj_pos": self.mujoco_simulation.get_target_pos().copy(),
            "obj_rot": self.mujoco_simulation.get_target_rot().copy(),
        }
        return goal_valid, goal

    def next_goal(self, random_state: RandomState, current_state: dict) -> dict:
        """
        Set goal position for each object and get goal dict.
        """
        goal_valid, goal_dict = self._update_simulation_for_next_goal(random_state)
        target_pos = goal_dict["obj_pos"]
        target_rot = goal_dict["obj_rot"]

        num_objects = self.mujoco_simulation.num_objects
        target_on_table = not self.mujoco_simulation.check_objects_off_table(
            target_pos[:num_objects]
        ).any()

        # Compute which of the goals is within the target placement area. Pad this to include
        # observations for up to max_num_objects (default to `1.0` for empty slots).
        # If an object is out of the placement, the corresponding position in the mask is 0.0.
        in_placement_area = self.mujoco_simulation.check_objects_in_placement_area(
            target_pos, margin=self.args.mask_margin, soft=self.args.soft_mask
        )
        assert in_placement_area.shape == (target_pos.shape[0],)

        # Create qpos for goal state: based on the current qpos but overwrite the object
        # positions to desired positions.
        num_objects = self.mujoco_simulation.num_objects
        qpos_goal = self.mujoco_simulation.qpos.copy()
        for i in range(num_objects):
            qpos_idx = self.mujoco_simulation.mj_sim.model.get_joint_qpos_addr(
                f"object{i}:joint"
            )[0]
            qpos_goal[qpos_idx: qpos_idx + 3] = target_pos[i].copy()
            qpos_goal[qpos_idx + 3: qpos_idx + 7] = rotation.euler2quat(target_rot[i])

        goal_invalid_reason: Optional[str] = None
        if not goal_valid:
            goal_invalid_reason = "Goal placement is invalid"
        elif not target_on_table:
            goal_invalid_reason = "Some goal objects are off the table."

        goal = {
            "obj_pos": target_pos.copy(),
            "obj_rot": target_rot.copy(),
            "qpos_goal": qpos_goal.copy(),
            "goal_valid": goal_valid and target_on_table,
            "goal_in_placement_area": in_placement_area.all(),
            "goal_objects_in_placement_area": in_placement_area.copy(),
            "goal_invalid_reason": goal_invalid_reason,
        }

        if self.args.rot_dist_type == "icp":
            goal["vertices"] = deepcopy(self.mujoco_simulation.get_target_vertices())
            goal["icp"] = [
                ICP(vertices, error_threshold=self.args.icp_error_threshold)
                for vertices in self.mujoco_simulation.get_target_vertices(
                    # Multiple by 2 because in theory max distance between
                    # a vertices and it's closest neighbor should be ~edge length / 2.
                    subdivide_threshold=self.args.icp_error_threshold
                    * 2
                )
            ]

            if self.args.icp_use_bbox_precheck:
                goal[
                    "bounding_box"
                ] = (
                    self.mujoco_simulation.get_target_bounding_boxes_in_table_coordinates().copy()
                )

        return goal

    def _randomize_goal_orientation(self, random_state: RandomState):
        rotated_quat = self._sample_next_goal_orientations(random_state)
        self.mujoco_simulation.set_target_quat(rotated_quat)
        self.mujoco_simulation.forward()

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        placement, is_valid = place_objects_in_grid(
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            random_state=random_state,
            max_num_trials=self.mujoco_simulation.max_placement_retry,
        )

        if not is_valid:
            # Fall back to random placement, which works better for envs with more irregular
            # objects (e.g. ycb-8 with no mesh normalization).
            return place_objects_with_no_constraint(
                self.mujoco_simulation.get_object_bounding_boxes(),
                self.mujoco_simulation.get_table_dimensions(),
                self.mujoco_simulation.get_placement_area(),
                max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
                max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
                random_state=random_state,
            )
        else:
            return placement, is_valid

    def _sample_next_goal_orientations(self, random_state: RandomState) -> np.ndarray:
        """ Sample goal orientation in quaternion """
        return self.rot_randomize_func(self.mujoco_simulation, random_state)

    def current_state(self) -> dict:
        """ Extract current cube goal state """
        state = {
            "obj_pos": self.mujoco_simulation.get_object_pos().copy(),
            "obj_rot": self.mujoco_simulation.get_object_rot().copy(),
            "qpos": self.mujoco_simulation.qpos.copy(),
        }

        if self.args.rot_dist_type == "icp":
            state["vertices"] = deepcopy(self.mujoco_simulation.get_object_vertices())
            state["icp"] = [
                ICP(vertices, error_threshold=self.args.icp_error_threshold)
                for vertices in self.mujoco_simulation.get_object_vertices(
                    # Multiple by 2 because in theory max distance between
                    # a vertices and it's closest neighbor should be ~edge length / 2.
                    subdivide_threshold=self.args.icp_error_threshold
                    * 2
                )
            ]

            if self.args.icp_use_bbox_precheck:
                state[
                    "bounding_box"
                ] = (
                    self.mujoco_simulation.get_target_bounding_boxes_in_table_coordinates().copy()
                )

        return state

    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        goal_pos = goal_state["obj_pos"]
        obj_pos = current_state["obj_pos"]

        if self.mujoco_simulation.num_objects == self.mujoco_simulation.num_groups:
            # All the objects are different.
            relative_obj_pos = goal_pos - obj_pos
            relative_obj_rot = self.rot_dist_func(goal_state, current_state)

        else:
            # per object relative pos & rot distance.
            rel_pos_dict = {}
            rel_rot_dict = {}

            def get_rel_rot(target_obj_id, curr_obj_id):
                group_goal_state = {"obj_rot": goal_state["obj_rot"][[target_obj_id]]}
                group_current_state = {
                    "obj_rot": current_state["obj_rot"][[curr_obj_id]]
                }

                if self.args.rot_dist_type == "icp":
                    group_goal_state["icp"] = [goal_state["icp"][target_obj_id]]
                    group_current_state["vertices"] = [
                        current_state["vertices"][curr_obj_id]
                    ]

                return self.rot_dist_func(group_goal_state, group_current_state)[0]

            for group_id, obj_group in enumerate(self.mujoco_simulation.object_groups):
                object_ids = obj_group.object_ids

                # Duplicated objects share the same group id.
                # Within each group we match objects with goals according to position in a greedy
                # fashion. Note that we ignore object rotation during matching.
                if len(object_ids) == 1:
                    object_id = object_ids[0]
                    rel_pos_dict[object_id] = goal_pos[object_id] - obj_pos[object_id]
                    rel_rot_dict[object_id] = get_rel_rot(object_id, object_id)

                else:
                    n = len(object_ids)

                    # find the optimal pair matching through greedy.
                    # TODO: may consider switching to `scipy.optimize.linear_sum_assignment`
                    assert obj_pos.shape == goal_pos.shape
                    dist = np.linalg.norm(
                        np.expand_dims(obj_pos[object_ids], 1)
                        - np.expand_dims(goal_pos[object_ids], 0),
                        axis=-1,
                    )
                    assert dist.shape == (n, n)

                    for _ in range(n):
                        i, j = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                        rel_pos_dict[object_ids[i]] = (
                            goal_pos[object_ids[j]] - obj_pos[object_ids[i]]
                        )
                        rel_rot_dict[object_ids[i]] = get_rel_rot(
                            object_ids[j], object_ids[i]
                        )
                        # once we select a pair of match (i, j), wipe out their distance info.
                        dist[i, :] = np.inf
                        dist[:, j] = np.inf

            assert (
                len(rel_pos_dict)
                == len(rel_rot_dict)
                == self.mujoco_simulation.num_objects
            )
            rel_pos = np.array(
                [rel_pos_dict[i] for i in range(self.mujoco_simulation.num_objects)]
            )
            rel_rot = np.array(
                [rel_rot_dict[i] for i in range(self.mujoco_simulation.num_objects)]
            )
            assert len(rel_pos.shape) == len(rel_rot.shape) == 2

            # padding zeros for the final output.
            relative_obj_pos = np.zeros(
                (self.mujoco_simulation.max_num_objects, rel_pos.shape[-1])
            )
            relative_obj_rot = np.zeros(
                (self.mujoco_simulation.max_num_objects, rel_rot.shape[-1])
            )
            relative_obj_pos[: rel_pos.shape[0]] = rel_pos
            relative_obj_rot[: rel_rot.shape[0]] = rel_rot

        # normalize angles
        relative_obj_rot = rotation.normalize_angles(relative_obj_rot)
        return {
            "obj_pos": relative_obj_pos.copy(),
            "obj_rot": relative_obj_rot.copy(),
        }

    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        relative_goal = self.relative_goal(goal_state, current_state)
        pos_distances = np.linalg.norm(relative_goal["obj_pos"], axis=-1)
        rot_distances = rotation.quat_magnitude(
            rotation.quat_normalize(rotation.euler2quat(relative_goal["obj_rot"]))
        )
        return {
            "relative_goal": relative_goal,
            "obj_pos": np.maximum(
                pos_distances + self.mujoco_simulation.goal_pos_offset, 0
            ),  # L2 dist
            "obj_rot": self.mujoco_simulation.goal_rot_weight
            * rot_distances,  # quat magnitude
        }
