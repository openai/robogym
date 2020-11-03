from robogym.envs.rearrange.simulation.mesh import (
    MeshRearrangeSim,
    MeshRearrangeSimParameters,
)
from robogym.mujoco.mujoco_xml import MujocoXML

CHESSBOARD_XML = """
<mujoco>
    <asset>
        <texture
            name="chessboard_texture" type="2d" builtin="checker"
            rgb1="0.616 0.341 0.106" rgb2="0.902 0.8 0.671"
            width="300" height="300">
        </texture>
        <material
            name="chessboard_mat" texture="chessboard_texture" texrepeat="{n_obj} {n_obj}"
            texuniform="false">
        </material>
    </asset>
    <worldbody>
        <body pos="1.3 0.75 0.4" name="chessboard">
                <geom
                    name="chessboard" size="0.16 0.16 0.001" type="box"
                    contype="0" conaffinity="0" material="chessboard_mat">
                </geom>
        </body>
    </worldbody>
</mujoco>
"""


class ChessboardRearrangeSim(MeshRearrangeSim):
    """
    Rearrange a chessboard.
    """

    @classmethod
    def make_world_xml(
        cls, *, mujoco_timestep: float, simulation_params=None, **kwargs
    ):
        if simulation_params is None:
            simulation_params = MeshRearrangeSimParameters()
        xml = super().make_world_xml(
            simulation_params=simulation_params,
            mujoco_timestep=mujoco_timestep,
            **kwargs
        )

        xml.append(
            MujocoXML.from_string(
                CHESSBOARD_XML.format(n_obj=simulation_params.num_objects // 2)
            )
        )

        return xml
