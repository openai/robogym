from typing import Dict

from robogym.robot.lib.parameter_configurer import ParameterConfigurer
from robogym.robot.shadow_hand.hand_interface import (
    ACTUATOR_JOINT_MAPPING,
    ACTUATORS,
    JOINTS,
)


class MuJoCoParameterManager(ParameterConfigurer):
    def __init__(self, sim):
        self.mj_sim = sim

    def set_parameters(self, actuator: str, assignments: Dict[str, float]):
        assert actuator in ACTUATORS

        actuator_id = self.mj_sim.model.actuator_name2id(actuator)

        # Based on http://www.mujoco.org/book/XMLreference.html#position.
        # second value in biasprm should be equal to -kp (first value of gainprm)
        # for actuators of "position" type.
        self.mj_sim.model.actuator_gainprm[actuator_id][0] = assignments[
            "actuator_gainprm_kp"
        ]
        self.mj_sim.model.actuator_gainprm[actuator_id][1] = assignments[
            "actuator_gainprm_ti"
        ]
        self.mj_sim.model.actuator_gainprm[actuator_id][2] = assignments[
            "actuator_gainprm_iclamp"
        ]
        self.mj_sim.model.actuator_gainprm[actuator_id][3] = assignments[
            "actuator_gainprm_td"
        ]
        self.mj_sim.model.actuator_gainprm[actuator_id][4] = assignments[
            "actuator_gainprm_dsmooth"
        ]
        self.mj_sim.model.actuator_gainprm[actuator_id][5] = assignments[
            "actuator_gainprm_error_deadband"
        ]

        self.mj_sim.model.actuator_forcerange[actuator_id][0] = -assignments[
            "actuator_forcerange"
        ]
        self.mj_sim.model.actuator_forcerange[actuator_id][1] = assignments[
            "actuator_forcerange"
        ]

        if self._has_spring_tendon(actuator):
            tendon = self._spring_tendon_name(actuator)
            tendon_id = self.mj_sim.model.tendon_name2id(tendon)
            self.mj_sim.model.tendon_stiffness[tendon_id] = assignments[
                "tendon_stiffness"
            ]
            self.mj_sim.model.tendon_lengthspring[tendon_id] = assignments[
                "tendon_lengthspring"
            ]
            self.mj_sim.model.tendon_range[tendon_id][1] = assignments["tendon_range"]

            for joint in ACTUATOR_JOINT_MAPPING[actuator]:
                geom = f"coupling_{joint}_pulley"
                geom_id = self.mj_sim.model.geom_name2id(geom)
                self.mj_sim.model.geom_size[geom_id][0] = assignments[
                    f"{joint}_tendon_geom_0"
                ]

        for joint in ACTUATOR_JOINT_MAPPING[actuator]:
            joint_id = self.mj_sim.model.joint_name2id(joint)
            self.mj_sim.model.dof_damping[joint_id] = assignments[
                f"{joint}_dof_damping"
            ]
            self.mj_sim.model.jnt_range[joint_id][0] = assignments[
                f"{joint}_jnt_range_0"
            ]
            self.mj_sim.model.jnt_range[joint_id][1] = assignments[
                f"{joint}_jnt_range_1"
            ]

    def current_parameters(self, actuator: str):
        assert actuator in ACTUATORS
        assignments = {}

        actuator_id = ACTUATORS.index(actuator)
        assignments["actuator_gainprm_kp"] = self.mj_sim.model.actuator_gainprm[
            actuator_id
        ][0]
        assignments["actuator_gainprm_ti"] = self.mj_sim.model.actuator_gainprm[
            actuator_id
        ][1]
        assignments["actuator_gainprm_iclamp"] = self.mj_sim.model.actuator_gainprm[
            actuator_id
        ][2]
        assignments["actuator_gainprm_td"] = self.mj_sim.model.actuator_gainprm[
            actuator_id
        ][3]
        assignments["actuator_gainprm_dsmooth"] = self.mj_sim.model.actuator_gainprm[
            actuator_id
        ][4]
        assignments[
            "actuator_gainprm_error_deadband"
        ] = self.mj_sim.model.actuator_gainprm[actuator_id][5]

        assignments["actuator_forcerange"] = self.mj_sim.model.actuator_forcerange[
            actuator_id
        ][1]

        if self._has_spring_tendon(actuator):
            tendon = self._spring_tendon_name(actuator)
            tendon_id = self.mj_sim.model.tendon_name2id(tendon)
            assignments["tendon_stiffness"] = self.mj_sim.model.tendon_stiffness[
                tendon_id
            ]
            assignments["tendon_lengthspring"] = self.mj_sim.model.tendon_lengthspring[
                tendon_id
            ]
            assignments["tendon_range"] = self.mj_sim.model.tendon_range[tendon_id][1]

            for joint in ACTUATOR_JOINT_MAPPING[actuator]:
                geom = f"coupling_{joint}_pulley"
                geom_id = self.mj_sim.model.geom_name2id(geom)
                assignments[f"{joint}_tendon_geom_0"] = self.mj_sim.model.geom_size[
                    geom_id
                ][0]

        for joint in ACTUATOR_JOINT_MAPPING[actuator]:
            joint_id = JOINTS.index(joint)
            assignments[f"{joint}_dof_damping"] = self.mj_sim.model.dof_damping[
                joint_id
            ]
            assignments[f"{joint}_jnt_range_0"] = self.mj_sim.model.jnt_range[joint_id][
                0
            ]
            assignments[f"{joint}_jnt_range_1"] = self.mj_sim.model.jnt_range[joint_id][
                1
            ]
        return assignments

    def parameter_bounds(self, actuator: str):
        assert actuator in ACTUATORS

        bounds = {}

        actuator_id = self.mj_sim.model.actuator_name2id(actuator)
        bounds["actuator_gainprm_kp"] = [
            0.25 * self.mj_sim.model.actuator_gainprm[actuator_id][0],
            4 * self.mj_sim.model.actuator_gainprm[actuator_id][0],
        ]
        bounds["actuator_gainprm_ti"] = [
            0.25 * self.mj_sim.model.actuator_gainprm[actuator_id][1],
            4 * self.mj_sim.model.actuator_gainprm[actuator_id][1] + 10.0,
        ]
        bounds["actuator_gainprm_iclamp"] = [
            0.25 * self.mj_sim.model.actuator_gainprm[actuator_id][2],
            4 * self.mj_sim.model.actuator_gainprm[actuator_id][2] + 10.0,
        ]
        bounds["actuator_gainprm_td"] = [
            0.25 * self.mj_sim.model.actuator_gainprm[actuator_id][3],
            4 * self.mj_sim.model.actuator_gainprm[actuator_id][3] + 0.1,
        ]
        bounds["actuator_gainprm_dsmooth"] = [
            0.0,
            0.2,
        ]
        bounds["actuator_gainprm_error_deadband"] = [
            0,
            0.03,
        ]
        # Note that force range min is equal to minus force range max,
        # thus we will be optimizing only one parameter.
        bounds["actuator_forcerange"] = [
            0.25 * self.mj_sim.model.actuator_forcerange[actuator_id][1],
            4 * self.mj_sim.model.actuator_forcerange[actuator_id][1],
        ]

        if self._has_spring_tendon(actuator):
            tendon = self._spring_tendon_name(actuator)
            tendon_id = self.mj_sim.model.tendon_name2id(tendon)
            bounds["tendon_stiffness"] = [
                0.25 * self.mj_sim.model.tendon_stiffness[tendon_id],
                4 * self.mj_sim.model.tendon_stiffness[tendon_id],
            ]
            bounds["tendon_lengthspring"] = [0.035, 0.075]
            bounds["tendon_range"] = [
                0.75 * self.mj_sim.model.tendon_range[tendon_id][1],
                1.25 * self.mj_sim.model.tendon_range[tendon_id][1],
            ]

            for joint in ACTUATOR_JOINT_MAPPING[actuator]:
                geom = f"coupling_{joint}_pulley"
                geom_id = self.mj_sim.model.geom_name2id(geom)
                bounds[f"{joint}_tendon_geom_0"] = [
                    0.5 * self.mj_sim.model.geom_size[geom_id][0],
                    1.5 * self.mj_sim.model.geom_size[geom_id][0],
                ]

        for joint in ACTUATOR_JOINT_MAPPING[actuator]:
            joint_id = self.mj_sim.model.joint_name2id(joint)
            bounds[f"{joint}_dof_damping"] = [0.01, 0.75]
            bounds[f"{joint}_jnt_range_0"] = [
                self.mj_sim.model.jnt_range[joint_id][0] - 0.25,
                self.mj_sim.model.jnt_range[joint_id][0] + 0.25,
            ]
            bounds[f"{joint}_jnt_range_1"] = [
                self.mj_sim.model.jnt_range[joint_id][1] - 0.25,
                self.mj_sim.model.jnt_range[joint_id][1] + 0.25,
            ]
        return bounds

    def _has_spring_tendon(self, actuator):
        return actuator in ["A_FFJ1", "A_MFJ1", "A_RFJ1", "A_LFJ1"]

    def _spring_tendon_name(self, actuator):
        assert self._has_spring_tendon(
            actuator
        ), f"{actuator} does not have spring tendon"
        return actuator.replace("A_", "")[:-2] + "T2"
