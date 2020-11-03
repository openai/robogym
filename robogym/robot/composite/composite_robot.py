import abc
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, cast

import numpy as np

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.robot_interface import RobotControlParameters, RobotObservation
from robogym.robot_env import Robot

OType = TypeVar("OType", bound=RobotObservation)


class CompositeRobot(Robot, Generic[OType], abc.ABC):
    """
    A composite robot that implements the Robot interface.
    """

    ROBOT_CLS: List[Type[Robot]] = []

    def __init__(
        self,
        simulation: SimulationInterface,
        solver_simulation: Optional[SimulationInterface],
        robot_control_params: RobotControlParameters,
        robot_prefix="robot0:",
        autostep=False,
    ):
        """
        :param simulation: simulation interface that supports all robots that belong to the class
        :param robot_control_params: Robot control parameters
        :param robot_prefix: prefix to add to the joint names while constructing the mujoco simulation
        :param autostep: when true, calls step() on the simulation whenever a control is set. this
        should only be used when the Robot is being controlled without a simulationrunner in the loop.
        """
        robot_cls = cast(List[Callable[..., Robot]], self.ROBOT_CLS)

        self.robots = [
            robot(
                simulation=simulation,
                solver_simulation=solver_simulation,
                robot_control_params=robot_control_params,
                robot_prefix=robot_prefix,
                autostep=False,
            )
            for robot in robot_cls
        ]

        self.autostep = autostep
        self.simulation = simulation
        self.action_space_partition = [
            len(robot.zero_control()) for robot in self.robots
        ]

    OBS_LABEL: List[str] = []

    OBS_CLS: Optional[Type[OType]] = None

    def get_name(self) -> str:
        return "composite-robot"

    def zero_control(self) -> np.ndarray:
        return np.concatenate([robot.zero_control() for robot in self.robots])

    @classmethod
    def actuators(cls) -> np.ndarray:
        return np.asarray([robot.actuators() for robot in cls.ROBOT_CLS])

    @classmethod
    def joints(cls) -> np.ndarray:
        return np.asarray([robot.joints() for robot in cls.ROBOT_CLS])

    def denormalize_position_control(
        self, position_control: np.ndarray, relative_action: bool = False,
    ) -> np.ndarray:
        offset = 0
        denormalized_control = []
        for i, robot in enumerate(self.robots):
            res = robot.denormalize_position_control(
                position_control=position_control[
                    offset: offset + self.action_space_partition[i]
                ],
                relative_action=relative_action,
            )
            denormalized_control.append(res)
            offset += self.action_space_partition[i]
        return np.concatenate(denormalized_control)

    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        return np.concatenate(
            [robot.actuator_ctrl_range_upper_bound() for robot in self.robots]
        )

    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        return np.concatenate(
            [robot.actuator_ctrl_range_lower_bound() for robot in self.robots]
        )

    def set_position_control(self, control: np.ndarray) -> None:
        offset = 0
        for i, robot in enumerate(self.robots):
            robot.set_position_control(
                control=control[offset: offset + self.action_space_partition[i]]
            )
            offset += self.action_space_partition[i]

        if self.autostep:
            self.simulation.mj_sim.step()

    def observe(self) -> OType:
        obs_cls = cast(Callable[..., OType], self.OBS_CLS)
        return obs_cls(
            robot_obs={k: v.observe() for k, v in zip(self.OBS_LABEL, self.robots)}
        )

    @classmethod
    def joint_positions_to_control(cls, joint_pos: np.ndarray):
        offset = 0
        control = []
        joint_space = [len(x) for x in cls.joints()]
        for i, robot in enumerate(cls.ROBOT_CLS):
            res = robot.joint_positions_to_control(
                joint_pos=joint_pos[offset: offset + joint_space[i]]
            )
            control.append(res)
            offset += joint_space[i]
        return np.concatenate(control)

    @property
    def max_position_change(self):
        raise NotImplementedError

    def reset(self) -> None:
        for robot in self.robots:
            robot.reset()

    def on_observations_updated(self, new_observations: Dict[str, Any]) -> None:
        """Event to notify the robot that new observations have been collected. See parents for more detailed
        documentation.

        Overridden to pass the event to the child robots.

        :param new_observations: New observations collected.
        """
        for robot in self.robots:
            robot.on_observations_updated(new_observations=new_observations)
