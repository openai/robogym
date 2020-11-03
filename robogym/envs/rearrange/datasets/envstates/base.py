from typing import Dict, List

import attr
import numpy as np
from numpy.random.mtrand import RandomState

from robogym.envs.rearrange.common.mesh import MeshRearrangeEnv
from robogym.envs.rearrange.datasets.objects.base import ObjectDataset
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig


@attr.s(auto_attribs=True)
class Envstate:
    """ Set of attributes that fully defines an environment state"""

    object_groups: List[ObjectGroupConfig]  # list of object groups for each objects
    object_dataset_names: List[str]  # object dataset names for each mesh
    object_ids: List[str]  # object id for each mesh in its object dataset
    init_pos: np.ndarray  # initial position of objects
    is_valid: bool  # whether placement of objects are valid
    init_quats: np.ndarray  # initial rotation of objects


class EnvstateDataset:
    """ Base class for environment state dataset """

    def __init__(
        self, object_datasets: Dict[str, ObjectDataset], random_state: RandomState
    ):
        """
        :param object_datasets: set of object datasets that are used to define environment states
            {dataset_name: dataset_object}.
        :param random_state: random state for envstate randomization.
        """
        self._random_state = random_state
        self.object_datasets = object_datasets
        self.envstate: Envstate = Envstate(
            object_groups=[],
            object_dataset_names=[],
            object_ids=[],
            init_pos=np.array([[]]),
            is_valid=False,
            init_quats=np.array([[]]),
        )
        self.initialized = False

    def reset(self, env: MeshRearrangeEnv):
        self._reset(env)
        self.initialized = True

    def _reset(self, env: MeshRearrangeEnv):
        """ Initialize self.envstate to fully define an environment's initial state """
        raise NotImplementedError

    def check_initialized(self):
        assert self.initialized, "dataset.reset(env) should be called before env reset"

    def _post_process_quat(
        self,
        quat: np.ndarray,
        object_groups: List[ObjectGroupConfig],
        mesh_object_dataset_names: List[str],
    ) -> np.ndarray:
        """
        Utility function for post processing object orientation based on object dataset
        specific logics
        """
        assert len(object_groups) == len(mesh_object_dataset_names)
        assert quat.shape[0] == np.sum(
            [grp.count for grp in object_groups]
        )  # num_objects

        result_quat = quat.copy()
        idx = 0
        for grp, dataset_name in zip(object_groups, mesh_object_dataset_names):
            dataset = self.object_datasets[dataset_name]
            result_quat[idx: idx + grp.count] = dataset.post_process_quat(
                result_quat[idx: idx + grp.count]
            )
            idx += grp.count
        return result_quat
