from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import PlacementArea
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal
from robogym.utils import rotation

MAX_RETRY = 1000


class DominoStateGoal(ObjectStateGoal):
    """
    This is a goal state generator for the dominos environment only, it does a lot of math to place the
    dominos (skewed blocks) into a circle arc.
    """

    @classmethod
    def _adjust_and_check_fit(
        cls,
        num_objects: int,
        target_bounding_boxes: np.ndarray,
        positions: np.ndarray,
        placement_area: PlacementArea,
        random_state: RandomState,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        This method will check if the current `target_bounding_boxes` and `positions` can fit in the table
        and if so, will return a new array of positions randomly sampled inside of the `placement_area`
        """
        width, height, _ = placement_area.size

        half_sizes = target_bounding_boxes[:, 1, :]

        max_x, max_y, _ = np.max(positions + half_sizes, axis=0)
        min_x, min_y, _ = np.min(positions - half_sizes, axis=0)

        size_x, size_y = max_x - min_x, max_y - min_y

        if size_x < width and size_y < height:
            # Sample a random offset of the "remaning area"
            delta_x = -min_x + random_state.random() * (width - size_x)
            delta_y = -min_y + random_state.random() * (height - size_y)

            return (
                True,
                positions + np.tile(np.array([delta_x, delta_y, 0]), (num_objects, 1)),
            )
        return False, None

    # Internal method
    def _create_new_domino_position_and_rotation(
        self, random_state: RandomState
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        This method will attempt at creating a new setup of dominos.
        The setup is to have the dominos be equally spaced across a circle arc

        :return: Tuple[proposed_positions, proposed_angles]
            proposed_positions: np.ndarray of shape (num_objects, 3) with proposed target positions, or None if placement fails
            proposed_angles: np.ndarray of shape (num_objects,) with proposed target angles, or None if placement fails
        """
        num_objects = self.mujoco_simulation.num_objects

        for _ in range(MAX_RETRY):
            # Offset that the whole domino chain is rotated by, this rotate the whole arc of dominos globally
            proposed_offset = random_state.random() * np.pi
            # Angle between the rotation of one domino and the next. Random angle offset between -pi/8 and pi/8
            proposed_delta = random_state.random() * (np.pi / 4.0) - (np.pi / 8.0)

            # Angles each domino will be rotated respectively
            proposed_angles = np.array(range(num_objects)) * proposed_delta + (
                proposed_offset + proposed_delta / 2
            )
            # Set target quat so that the computation for `get_target_bounding_boxes` is correct
            self._set_target_quat(num_objects, proposed_angles)

            angles_between_objects = (
                np.array(range(1, 1 + num_objects)) * proposed_delta + proposed_offset
            )

            object_distance = (
                self.mujoco_simulation.simulation_params.object_size
                * self.mujoco_simulation.simulation_params.domino_distance_mul
            )

            x = np.cumsum(np.cos(angles_between_objects)) * object_distance
            y = np.cumsum(np.sin(angles_between_objects)) * object_distance

            # Proposed positions
            proposed_positions = np.zeros((num_objects, 3))

            # Copy the z axis values:
            target_bounding_boxes = (
                self.mujoco_simulation.get_target_bounding_boxes()
            )  # List[obj_pos, obj_size]
            proposed_positions[:, 2] = target_bounding_boxes[:, 1, 2]

            for target_idx in range(num_objects):
                # First target will be at (0, 0) and the remaning targets will be offset from that
                if target_idx > 0:
                    proposed_positions[target_idx] += np.array(
                        [x[target_idx - 1], y[target_idx - 1], 0]
                    )

            is_valid, proposed_positions = self._adjust_and_check_fit(
                num_objects,
                target_bounding_boxes,
                proposed_positions,
                self.mujoco_simulation.get_placement_area(),
                random_state,
            )
            if is_valid and proposed_positions is not None:
                return proposed_positions, proposed_angles

        # Mark failure to fit goal positions
        return None, None

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        num_objects = self.mujoco_simulation.num_objects

        (
            proposed_positions,
            proposed_angles,
        ) = self._create_new_domino_position_and_rotation(random_state)

        if proposed_positions is None or proposed_angles is None:
            # The `False` will show this is not a valid goal position
            return np.zeros((num_objects, 3)), False

        # Add the needed offsets
        (
            table_pos,
            table_size,
            table_height,
        ) = self.mujoco_simulation.get_table_dimensions()
        placement_area = self.mujoco_simulation.get_placement_area()

        start_x, start_y, _ = table_pos - table_size
        start_x += placement_area.offset[0]
        start_y += placement_area.offset[1]

        return_value = proposed_positions + np.tile(
            [start_x, start_y, table_height], (num_objects, 1)
        )

        return return_value, True

    def _set_target_quat(self, num_objects: int, proposed_angles: np.ndarray) -> None:
        proposed_quaternions = rotation.quat_from_angle_and_axis(
            angle=proposed_angles, axis=np.array([[0, 0, 1.0]] * num_objects)
        )

        self.mujoco_simulation.set_target_quat(proposed_quaternions)
        self.mujoco_simulation.forward()
