from robogym.envs.rearrange.common.utils import geom_ids_of_body
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.gripper.mujoco.mujoco_robotiq_gripper import MujocoRobotiqGripper
from robogym.robot.robot_interface import RobotControlParameters


class ArmSimulationInterface(SimulationInterface):
    """
    Creates a SimulationInterface with a rearrange-compatible robot-gripper and a
    table setup. Subclass this and implement make_objects_xml() to create other tasks.
    """

    DEFAULT_RENDER_SIZE = 100

    BASE_XML = "robot/ur16e/base.xml"

    def __init__(
        self, sim, robot_control_params: RobotControlParameters,
    ):
        super().__init__(sim)

        self.register_joint_group("robot", prefix=["robot0:"])
        self.register_joint_group(
            "gripper", prefix=["robot0:r_gripper", "robot0:l_gripper"]
        )
        self.control_param = robot_control_params

        self.enable_pid()

        # initialize a gripper in sim so that it can be used to sync state if we need to.
        self._gripper = MujocoRobotiqGripper(
            simulation=self, robot_control_params=robot_control_params, autostep=False
        )

        # Hide mocap since not very helpful and clutters vision.
        mocap_id = self.mj_sim.model.body_name2id("robot0:mocap")
        mocap_geom_start_id = self.mj_sim.model.body_geomadr[mocap_id]
        mocap_geom_end_id = (
            mocap_geom_start_id + self.mj_sim.model.body_geomnum[mocap_id]
        )
        for geom_id in range(mocap_geom_start_id, mocap_geom_end_id):
            self.mj_sim.model.geom_rgba[geom_id, :] = 0.0

        self.geom_ids = []
        self.gripper_bodies = [
            "robot0:gripper_base",
            "left_gripper",
            "left_inner_follower",
            "left_outer_driver",
            "right_gripper",
            "right_inner_follower",
            "right_outer_driver",
        ]

        # Get the geom ids of all the bodies that make up the gripper
        for gripper_body in self.gripper_bodies:
            self.geom_ids.extend(geom_ids_of_body(self.mj_sim, gripper_body))

    @classmethod
    def build(
        cls,
        robot_control_params: RobotControlParameters,
        n_substeps=40,
        mujoco_timestep=0.001,
    ):
        xml = cls.make_world_xml(
            contact_params=dict(njmax=200, nconmax=200, nuserdata=200),
            mujoco_timestep=mujoco_timestep,
        )

        xml = ArmSimulationInterface.make_robot_xml(xml, robot_control_params)

        return cls(
            xml.build(nsubsteps=n_substeps), robot_control_params=robot_control_params,
        )

    @classmethod
    def make_world_xml(cls, *, contact_params: dict, mujoco_timestep: float, **kwargs):
        return (
            MujocoXML.parse(cls.BASE_XML)
            .set_objects_attr(tag="option", timestep=mujoco_timestep)
            .set_objects_attr(tag="size", **contact_params)
            .add_default_compiler_directive()
        )

    @classmethod
    def make_robot_xml(cls, xml, robot_control_params):
        if robot_control_params.is_joint_actuated():
            # Modifying xml is required because setting eq_active only was not enough to fully
            # disable the mocap weld constraint. In my tests, setting eq_active to false would
            # disable the constraint, but somehow the arm would not move when the joints were
            # commanded. Removing from xml here seems to have the right effect.
            xml.remove_objects_by_name("mocap_weld")
            # Also add the actuations that are removed in the xml by default (since TCP does
            # not need them).
            joint_subdir = robot_control_params.arm_joint_calibration_path
            xml.append(
                MujocoXML.parse(
                    f"robot/ur16e/jointspec/calibrations/{joint_subdir}/ur16e_ik_class.xml"
                )
            )
            xml.append(
                MujocoXML.parse(
                    f"robot/ur16e/jointspec/calibrations/{joint_subdir}/joint_actuations.xml"
                )
            )
        else:
            # If not joint control mode or ik solver mode, use mocap defaults for joint parameters
            xml.append(MujocoXML.parse("robot/ur16e/jointspec/ur16e_mocap_class.xml"))

        # Add gripper actuators now (after joint actuators if required).
        xml.append(MujocoXML.parse("robot/ur16e/gripper_actuators.xml"))

        return xml

    @property
    def gripper(self):
        return self._gripper

    def render(
        self,
        width=DEFAULT_RENDER_SIZE,
        height=DEFAULT_RENDER_SIZE,
        *,
        camera_name="vision_cam_front",
        depth=False,
        mode="offscreen",
        device_id=-1,
    ):
        data = super().render(
            width=width,
            height=height,
            camera_name=camera_name,
            depth=depth,
            mode=mode,
            device_id=device_id,
        )
        # original image is upside-down, so flip it
        return data[::-1, :, :]

    def get_gripper_table_contact(self) -> bool:
        """
        Determine if any part of the gripper is touching the table by checking if there
        is a collision between the table_collision_plane id and any gripper geom id.
        """

        contacts = []
        gripper_table_contact = False

        # Sweep through all mj_sim contacts
        for i in range(self.mj_sim.data.ncon):
            c = self.mj_sim.data.contact[i]

            # Check if any of the contacts involve at gripper geom id, append them to contacts:
            if c.geom1 in self.geom_ids:
                contacts.append(c.geom2)
            elif c.geom2 in self.geom_ids:
                contacts.append(c.geom1)

            # Check if any of the contacts correspond to the `table` id:
            for contact in contacts:
                contact_name = self.mj_sim.model.geom_id2name(contact)
                if contact_name == "table_collision_plane":
                    gripper_table_contact = True

        return gripper_table_contact
