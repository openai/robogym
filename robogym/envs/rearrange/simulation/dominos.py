import attr
import numpy as np

from robogym.envs.rearrange.common.utils import (
    get_block_bounding_box,
    make_blocks_and_targets,
)
from robogym.envs.rearrange.simulation.base import (
    RearrangeSimParameters,
    RearrangeSimulationInterface,
)
from robogym.randomization.env import build_randomizable_param


@attr.s(auto_attribs=True)
class DominosRearrangeSimParameters(RearrangeSimParameters):
    # How "skewed" the domino is. A bigger value corresponds to a thinner and taller domino.
    domino_eccentricity: float = build_randomizable_param(1.5, low=1.0, high=4.5)

    # The proportion compared to `object_size` to distance the dominos from each other
    domino_distance_mul: float = build_randomizable_param(4, low=2.0, high=5.0)

    num_objects: int = build_randomizable_param(5, low=1, high=8)


class DominosRearrangeSim(RearrangeSimulationInterface[DominosRearrangeSimParameters]):
    """
    Move around a dominos of different colors on the table.

    Similar to BlockRearrangeEnv, but with dominos in a circle arc position.
    """

    @classmethod
    def make_objects_xml(cls, xml, simulation_params: DominosRearrangeSimParameters):
        skewed_object_size = simulation_params.object_size * np.array(
            [
                1 / simulation_params.domino_eccentricity,
                1,
                1 * simulation_params.domino_eccentricity,
            ]
        )
        return make_blocks_and_targets(
            simulation_params.num_objects, skewed_object_size
        )

    def _get_bounding_box(self, object_name):
        return get_block_bounding_box(self.mj_sim, object_name)

    @property
    def domino_eccentricity(self):
        return self.simulation_params.domino_eccentricity
