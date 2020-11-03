from typing import Any, Dict, List

import numpy as np
from numpy.random.mtrand import RandomState


class ObjectDataset:
    """ Base class for object dataset """

    def __init__(self, random_state: RandomState, mesh_scale: float = 1.0):
        self._random_state = random_state
        self.mesh_scale = mesh_scale

        # Set of all objects that can be sampled from this datasetset. {object_id: path to object}
        self.objects: Dict[str, Any] = self.get_objects()
        # List of all object ids
        self.object_ids: List[str] = list(self.objects.keys())

    def get_objects(self) -> Dict[str, Any]:
        """ Return a dictionary of {object_id: pointer to the object meshes}"""
        raise NotImplementedError

    def get_mesh_list(self, object_ids: List[str]) -> List[List[str]]:
        """ Return a list of (a list of mesh file paths) for a list of objects """
        meshes = []
        for object_id in object_ids:
            meshes.append(self.get_mesh(object_id))
        return meshes

    def get_mesh(self, object_id: str) -> List[str]:
        """ Return a list of mesh file paths for an object """
        raise NotImplementedError

    def sample(self, num_groups: int) -> List[str]:
        """ Sample object ids for this dataset """
        raise NotImplementedError

    @classmethod
    def post_process_quat(cls, quat: np.ndarray) -> np.ndarray:
        """
        Apply object dataset specific logic to post process object's initial rotation

        This function can be useful if all meshes in some object dataset has weird default
        orientations.
        """
        return quat
