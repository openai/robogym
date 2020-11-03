from robogym.envs.rearrange.common.utils import get_block_bounding_box, make_target
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.mujoco.mujoco_xml import ASSETS_DIR, MujocoXML


class WordBlocksSim(RearrangeSimulationInterface):
    """
    reuse the OPENAI character meshes (used for block cube)
    """

    @classmethod
    def make_objects_xml(cls, xml, simulation_params):
        def get_cube_mesh(c) -> str:
            return f"""<mesh name="object:letter_{c}" file="{ASSETS_DIR}/stls/openai_cube/{c}.stl" />"""

        def get_cube_body(idx, c) -> str:
            color = "0.71 0.61 0.3"
            letter_material = "cube:letter_white"

            return f"""
<mujoco>
    <worldbody>
        <body name="object{idx}">
            <joint name="object{idx}:joint" type="free"/>
            <geom name="object{idx}" size="0.0285 0.0285 0.0285" type="box" rgba="{color} 1" />
            <body name="object:letter_{c}:face_top" pos="0 0 0.0285">
                <geom name="object:letter_{c}" pos="0 0 -0.0009" quat="0.499998 0.5 -0.500002 0.5" type="mesh" material="{letter_material}" mesh="object:letter_{c}" />
            </body>
        </body>
    </worldbody>
</mujoco>
"""

        letter_meshes = "\n".join([get_cube_mesh(c) for c in "OPENAI"])

        assets_xml = f"""
<mujoco>
    <asset>
        <material name="cube:top_background" specular="1" shininess="0.3" />
        <material name="cube:letter" specular="1" shininess="0.3" rgba="0. 0. 1 1"/>
        <material name="cube:letter_white" specular="1" shininess="0.3" rgba="1 1 1 1"/>
        {letter_meshes}
    </asset>
</mujoco>
        """

        xml.append(MujocoXML.from_string(assets_xml))

        xmls = []

        for idx, c in enumerate("OPENAI"):
            if idx >= simulation_params.num_objects:
                break
            object_xml = MujocoXML.from_string(get_cube_body(idx, c))
            xmls.append((object_xml, make_target(object_xml)))

        return xmls

    def _get_bounding_box(self, object_name):
        return get_block_bounding_box(self.mj_sim, object_name)
