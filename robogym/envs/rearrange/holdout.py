import os
from typing import Dict, List, Optional, cast

import attr
import numpy as np

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.goals.holdout_object_state import (
    HoldoutGoalArgs,
    HoldoutObjectStateGoal,
)
from robogym.envs.rearrange.holdouts import STATE_DIR
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.holdout import (
    HoldoutRearrangeSim,
    HoldoutRearrangeSimParameters,
)
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class HoldoutRearrangeEnvConstants(RearrangeEnvConstants):

    # Path to file storing initial state of objects.
    # If not specified, initial state will be randomly sampled.
    initial_state_path: Optional[str] = None

    goal_args: HoldoutGoalArgs = build_nested_attr(HoldoutGoalArgs)

    randomize_target: bool = False


@attr.s(auto_attribs=True)
class HoldoutRearrangeEnvParameters(RearrangeEnvParameters):
    simulation_params: HoldoutRearrangeSimParameters = build_nested_attr(
        HoldoutRearrangeSimParameters
    )

    # Hold out arg should use explicitly defined material without randomization.
    material_names: Optional[List[str]] = attr.ib(default=cast(List[str], []))

    @material_names.validator
    def validate_material_names(self, _, value):
        assert not value, (
            "Specifying material names for holdout in parameters is not supported. "
            "Please specify material in jsonnet config directly."
        )


class HoldoutRearrangeEnv(
    RearrangeEnv[
        HoldoutRearrangeEnvParameters,
        HoldoutRearrangeEnvConstants,
        HoldoutRearrangeSim,
    ]
):
    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:

        # Create dummy object groups based on task object config so that reward
        # function can take duplicated objects in holdouts into consideration.

        object_groups = []
        num_objects = self.parameters.simulation_params.num_objects
        object_id = 0
        for c in self.parameters.simulation_params.task_object_configs[:num_objects]:
            object_group = ObjectGroupConfig(count=c.count)

            # Set up object ids
            object_group.object_ids = list(range(object_id, object_id + c.count))
            object_id += c.count

            object_groups.append(object_group)
        return object_groups

    def _sample_group_attributes(self, num_groups: int) -> Dict[str, list]:
        # We don't set random attributes for object groups
        return {}

    def _apply_object_colors(self):
        # We don't apply customized object colors.
        pass

    def _apply_object_size_scales(self):
        # We don't apply customized object size scaling.
        pass

    def _randomize_object_initial_states(self):
        if self.constants.initial_state_path:
            initial_state = np.load(
                os.path.join(STATE_DIR, self.constants.initial_state_path)
            )
            self.mujoco_simulation.set_object_pos(
                initial_state["obj_pos"][: self.mujoco_simulation.num_objects]
            )
            self.mujoco_simulation.set_object_quat(
                initial_state["obj_quat"][: self.mujoco_simulation.num_objects]
            )
            self.mujoco_simulation.forward()
        else:
            super()._randomize_object_initial_states()

    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        if constants.randomize_target:
            return super().build_goal_generation(constants, mujoco_simulation)
        else:
            return HoldoutObjectStateGoal(mujoco_simulation, args=constants.goal_args)


make_env = HoldoutRearrangeEnv.build
