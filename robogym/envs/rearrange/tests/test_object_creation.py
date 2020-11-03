import numpy as np
from numpy.testing import assert_allclose

from robogym.envs.rearrange.common.utils import (
    get_mesh_bounding_box,
    make_block,
    make_blocks_and_targets,
)
from robogym.envs.rearrange.simulation.composer import RandomMeshComposer
from robogym.mujoco.mujoco_xml import MujocoXML


def _get_default_xml():
    xml_source = """
    <mujoco>
      <asset>
        <material name="block_mat" specular="0" shininess="0.5" reflectance="0" rgba="1 0 0 1"></material>
      </asset>
    </mujoco>
    """
    xml = MujocoXML.from_string(xml_source)
    return xml


def test_mesh_composer():
    for path in [
        None,
        RandomMeshComposer.GEOM_ASSET_PATH,
        RandomMeshComposer.GEOM_ASSET_PATH,
    ]:
        composer = RandomMeshComposer(mesh_path=path)
        for num_geoms in range(1, 6):
            xml = _get_default_xml()
            composer.reset()
            xml.append(composer.sample("object0", num_geoms, object_size=0.05))
            sim = xml.build()
            assert len(sim.model.geom_names) == num_geoms
            pos, size = get_mesh_bounding_box(sim, "object0")
            assert np.isclose(np.max(size), 0.05)
            pos2, size2 = composer.get_bounding_box(sim, "object0")
            assert np.allclose(pos, pos2)
            assert np.allclose(size, size2)


def test_block_object():
    xml = _get_default_xml()
    xml.append(make_block("object0", object_size=np.ones(3) * 0.05))
    sim = xml.build()
    assert len(sim.model.geom_size) == 1
    assert_allclose(sim.model.geom_size, 0.05)


def test_blocks_and_targets():
    xml = _get_default_xml()
    for obj_xml, target_xml in make_blocks_and_targets(num_objects=5, block_size=0.05):
        xml.append(obj_xml)
        xml.append(target_xml)

    sim = xml.build()
    assert len(sim.model.geom_size) == 10
    assert_allclose(sim.model.geom_size, 0.05)
