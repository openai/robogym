from typing import List

from robogym.envs.rearrange.blocks import BlockRearrangeEnv
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig


class DuplicateBlockRearrangeEnv(BlockRearrangeEnv):
    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        """
        Create one group of block objects with a random color.
        Overwrite the object groups info to contain only one group for all the blocks.
        """
        object_groups = super()._sample_random_object_groups()

        num_objects = self.parameters.simulation_params.num_objects
        first_object_group = object_groups[0]
        first_object_group.count = num_objects
        first_object_group.object_ids = list(range(num_objects))
        object_groups = [first_object_group]
        return object_groups


make_env = DuplicateBlockRearrangeEnv.build
