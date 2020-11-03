import numpy as np

# noinspection PyUnresolvedReferences
from robogym.mujoco.modifiers.timestep import Modifier


# noinspection PyAttributeOutsideInit
class PerpendicularCubeSizeModifier(Modifier):
    """ Modify size of a "perpendicular cube" """

    def __init__(self, prefix):
        super().__init__()

        self.body_name_prefix = f"{prefix}cubelet:"
        self.mesh_name = f"{prefix}rounded_cube"

    def initialize(self, sim):
        super().initialize(sim)

        cubelet_body_names = [
            x for x in self.sim.model.body_names if x.startswith(self.body_name_prefix)
        ]

        self.cubelet_body_ids = np.array(
            [self.sim.model.body_name2id(x) for x in cubelet_body_names]
        )

        cubelet_geom_names = [
            x for x in self.sim.model.geom_names if x.startswith(self.body_name_prefix)
        ]

        self.cubelet_geom_ids = np.array(
            [self.sim.model.geom_name2id(x) for x in cubelet_geom_names]
        )

        cube_mesh_id = self.sim.model.mesh_name2id(self.mesh_name)

        self.cube_vert_adr = self.sim.model.mesh_vertadr[cube_mesh_id]
        self.cube_vert_num = self.sim.model.mesh_vertnum[cube_mesh_id]

        self.original_cube_body_pos = self.sim.model.body_pos[
            self.cubelet_body_ids
        ].copy()

        self.original_mesh_verts = (
            self.sim.model.mesh_vert[
                self.cube_vert_adr: self.cube_vert_adr + self.cube_vert_num
            ]
        ).copy()

        self.original_geom_rbound = self.sim.model.geom_rbound[
            self.cubelet_geom_ids
        ].copy()

    def __call__(self, cube_size_multiplier):
        self.sim.model.body_pos[self.cubelet_body_ids] = (
            self.original_cube_body_pos * cube_size_multiplier
        )

        self.sim.model.mesh_vert[
            self.cube_vert_adr: self.cube_vert_adr + self.cube_vert_num
        ] = (self.original_mesh_verts * cube_size_multiplier)

        self.sim.model.geom_rbound[self.cubelet_geom_ids] = (
            self.original_geom_rbound * cube_size_multiplier
        )


# noinspection PyAttributeOutsideInit
class LockedCubeSizeModifier(Modifier):
    """ Modify size of a "locked cube" """

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def initialize(self, sim):
        super().initialize(sim)

        self.body_ids = [
            self.sim.model.body_name2id(x)
            for x in self.sim.model.body_names
            if x.startswith(self.prefix)
        ]

        self.original_body_pos = self.sim.model.body_pos[self.body_ids].copy()

        self.geom_ids = [
            self.sim.model.geom_name2id(x)
            for x in self.sim.model.geom_names
            if x.startswith(self.prefix)
        ]
        self.original_geom_size = self.sim.model.geom_size[self.geom_ids].copy()

    def __call__(self, cube_size_multiplier):
        self.sim.model.body_pos[self.body_ids] = (
            self.original_body_pos * cube_size_multiplier
        )
        self.sim.model.geom_size[self.geom_ids] = (
            self.original_geom_size * cube_size_multiplier
        )
