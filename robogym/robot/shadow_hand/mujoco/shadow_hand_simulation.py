import numpy as np

from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.shadow_hand.mujoco.mujoco_shadow_hand import MuJoCoShadowHand


class ShadowHandSimulation(SimulationInterface):
    """
    MuJoCo simulation containing only the shadow hand
    """

    # Robot hand xml
    HAND_XML = "robot/shadowhand/main.xml"
    # Just a floor
    FLOOR_XML = "floor/basic_floor.xml"
    # XML with default light
    LIGHT_XML = "light/default.xml"

    @classmethod
    def build(
        cls, n_substeps: int = 10, timestep: float = 0.008, name_prefix: str = "robot0:"
    ):
        """
        Construct MjSim object for this simulation

        :param name_prefix - to append to names of all objects in the MuJoCo model of the hand;
            by default no prefix is appended.
        """
        xml = MujocoXML()
        xml.add_default_compiler_directive()

        max_contacts_params = dict(njmax=2000, nconmax=200)

        xml.append(
            MujocoXML.parse(cls.FLOOR_XML).set_named_objects_attr(
                "floor", tag="body", pos=[1, 1, 0]
            )
        )

        xml.append(
            MujocoXML.parse(cls.HAND_XML)
            .add_name_prefix(name_prefix)
            .set_objects_attr(tag="size", **max_contacts_params)
            .set_objects_attr(tag="option", timestep=timestep)
            .set_named_objects_attr(
                f"{name_prefix}hand_mount",
                tag="body",
                pos=[1.0, 1.25, 0.15],
                euler=[np.pi / 2, 0, np.pi],
            )
            .remove_objects_by_name(f"{name_prefix}annotation:outer_bound")
            # Remove hand base free joint so that hand is immovable
            .remove_objects_by_name(f"{name_prefix}hand_base")
        )

        xml.append(MujocoXML.parse(cls.LIGHT_XML))

        return cls(sim=xml.build(nsubsteps=n_substeps), hand_prefix=name_prefix)

    def __init__(self, sim, hand_prefix="robot0:"):
        super().__init__(sim)
        self.enable_pid()
        self.shadow_hand = MuJoCoShadowHand(self, hand_prefix=hand_prefix)
