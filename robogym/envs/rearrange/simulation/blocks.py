import attr

from robogym.envs.rearrange.common.utils import (
    get_block_bounding_box,
    make_blocks_and_targets,
)
from robogym.envs.rearrange.simulation.base import (
    RearrangeSimParameters,
    RearrangeSimulationInterface,
)


@attr.s(auto_attribs=True)
class BlockRearrangeSimParameters(RearrangeSimParameters):
    # Appearance of the block. 'standard' blocks have plane faces without any texture or mark.
    # 'openai' blocks have ['O', 'P', 'E', 'N', 'A', 'I'] in each face.
    block_appearance: str = attr.ib(
        default="standard", validator=attr.validators.in_(["standard", "openai"])
    )


class BlockRearrangeSim(RearrangeSimulationInterface[BlockRearrangeSimParameters]):
    """
    Move around a blocks of different colors on the table.
    """

    @classmethod
    def make_objects_xml(cls, xml, simulation_params: BlockRearrangeSimParameters):
        return make_blocks_and_targets(
            simulation_params.num_objects,
            simulation_params.object_size,
            appearance=simulation_params.block_appearance,
        )

    def _get_bounding_box(self, object_name):
        return get_block_bounding_box(self.mj_sim, object_name)
