from typing import Any, Dict, List

import attr

from robogym.envs.rearrange.common.mesh import (
    MeshRearrangeEnv,
    MeshRearrangeEnvConstants,
    MeshRearrangeEnvParameters,
)
from robogym.envs.rearrange.datasets.envstates.utils import get_envstate_datasets
from robogym.envs.rearrange.datasets.objects.utils import get_object_datasets
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim


@attr.s(auto_attribs=True)
class MixtureRearrangeEnvConstants(MeshRearrangeEnvConstants):

    # Set of object datasets used for constructing environment states.
    # {object dataset name: object dataset config}
    object_config: Dict[str, Dict[str, Any]] = {
        "ycb": {
            "function": "robogym.envs.rearrange.datasets.objects.local_mesh:create",
            "args": {"mesh_dirname": "ycb"},
        },
        "geom": {
            "function": "robogym.envs.rearrange.datasets.objects.local_mesh:create",
            "args": {"mesh_dirname": "geom"},
        },
    }

    # Sef of environment state datasets that are used to sample environment states. The
    # environment state datasets use object datasets defined by object_config.
    # {envstate dataset name: dataset config}
    dataset_config: Dict[str, Dict[str, Any]] = {
        "ycb_dataset": {
            "function": "robogym.envs.rearrange.datasets.envstates.random:create",
            "args": {"object_sample_prob": {"ycb": 1.0}},
        },
        "geom_dataset": {
            "function": "robogym.envs.rearrange.datasets.envstates.random:create",
            "args": {"object_sample_prob": {"geom": 1.0}},
        },
        "mixed_dataset": {
            "function": "robogym.envs.rearrange.datasets.envstates.random:create",
            "args": {"object_sample_prob": {"ycb": 0.5, "geom": 0.5}},
        },
    }

    # environment state dataset level sampling probability.
    # {envstate dataset name: probability of sampling from this dataset}
    dataset_sampling_config: dict = {
        "ycb_dataset": 0.3,
        "geom_dataset": 0.3,
        "mixed_dataset": 0.4,
    }


class MixtureRearrangeEnv(
    MeshRearrangeEnv[
        MeshRearrangeEnvParameters, MixtureRearrangeEnvConstants, MeshRearrangeSim
    ]
):
    """
    Rearrange environment using mixture of dataset to define an initial state distribution.
    """

    def initialize(self):
        super().initialize()
        self.object_datasets = get_object_datasets(
            self.constants.object_config, self._random_state
        )
        self.datasets = get_envstate_datasets(
            self.constants.dataset_config, self.object_datasets, self._random_state
        )

        # Name of environment state datasets
        self.dataset_ids = sorted(list(self.constants.dataset_sampling_config))

        # Probability for sampling from each environment state datasets
        self.dataset_prob = [
            self.constants.dataset_sampling_config[dataset_id]
            for dataset_id in self.dataset_ids
        ]

        # Dataset that will be used for sampling an environment state
        self.cur_dataset = self._sample_dataset()

    def _sample_dataset(self):
        """ Sample an environment state dataset """
        dataset_id = self.dataset_ids[
            self._random_state.choice(len(self.dataset_ids), p=self.dataset_prob)
        ]
        return self.datasets[dataset_id]

    def _reset(self):
        """ Reset environment state

        This function resets envstate dataset and then use the dataset state for resetting the
        environment. environment state dataset may randomize on the fly or load previously saved
        environment states from storage.
        """
        self.cur_dataset = self._sample_dataset()
        self.cur_dataset.reset(self)
        super()._reset()

    #######################################################################################
    # Override envstate randomization function in environment to make the dataset fully
    # determine the environment states

    def _sample_attributed_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        """ This function includes sampling mesh, scales, and colors """
        assert not dedupe_objects, "Mixture dataset always supports duplicated objects"
        return self.cur_dataset.envstate.object_groups

    def _sample_object_meshes(self, num_groups: int) -> List[List[str]]:
        # This function is not necessary because we will directly return object groups
        pass

    def _generate_object_placements(self):
        self.cur_dataset.check_initialized()
        return self.cur_dataset.envstate.init_pos, self.cur_dataset.envstate.is_valid

    def _sample_object_initial_rotations(self):
        self.cur_dataset.check_initialized()
        return self.cur_dataset.envstate.init_quats


make_env = MixtureRearrangeEnv.build
