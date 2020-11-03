from numpy.random.mtrand import RandomState

from robogym.envs.rearrange.common.utils import find_meshes_by_dirname
from robogym.envs.rearrange.datasets.objects.base import ObjectDataset


class LocalMeshObjectDataset(ObjectDataset):
    """
    Class of object dataset loading mesh files for objects from local directory.

    mesh files are loaded from robogym/assets/stls/{mesh_dirname}
    e.g.
    * mesh_dirname == 'ycb': load ycb object dataset
    * mesh_dirname == 'geom': load geom object dataset
    """

    def __init__(
        self, random_state: RandomState, mesh_dirname: str, mesh_scale: float = 1.0
    ):
        """
        :param mesh_dirname: local directory name (path: robogym/assets/stls/{mesh_dirname}) to
            load object mesh files from
        """
        self.mesh_dirname = mesh_dirname
        super().__init__(random_state=random_state, mesh_scale=mesh_scale)

    def get_mesh(self, object_id: str):
        return self.objects[object_id]

    def get_objects(self):
        return find_meshes_by_dirname(self.mesh_dirname)

    def sample(self, num_groups: int):
        indices = self._random_state.choice(len(self.object_ids), size=num_groups)
        return [self.object_ids[i] for i in indices]


create = LocalMeshObjectDataset
