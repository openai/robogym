import numpy as np
from numpy.random import RandomState

from robogym.envs.dactyl.reach import ReachSimulation
from robogym.goal.goal_generator import GoalGenerator
from robogym.robot.shadow_hand.hand_forward_kinematics import FINGERTIP_SITE_NAMES
from robogym.utils.dactyl_utils import actuated_joint_range


class FingertipPosGoal(GoalGenerator):
    """
    Goal generation to sample random qpos within actuator control range.
    """

    def __init__(
        self, mujoco_simulation: ReachSimulation, goal_simulation: ReachSimulation
    ):
        """
        Create new FingertipPosGoal object
        """
        self.mujoco_simulation = mujoco_simulation
        self.goal_simulation = goal_simulation
        self.goal_joint_pos = mujoco_simulation.shadow_hand.observe().joint_positions()

        super().__init__()

    def next_goal(self, random_state: RandomState, current_state: dict) -> dict:
        """
        Goal is defined as fingertip position.

        We sample next goal by sampling actuator control within control range then use
        forward kinematic to calculate fingertip position.
        """

        sim = self.mujoco_simulation.mj_sim
        goal_sim = self.goal_simulation.mj_sim

        # We need to update control range and joint range for goal simulation because
        # they can be changed by randomizers.
        goal_sim.model.jnt_range[:] = sim.model.jnt_range
        goal_sim.model.actuator_ctrlrange[:] = sim.model.actuator_ctrlrange

        # Sample around current pose of the fingers in joint space.
        joint_limits = actuated_joint_range(sim)
        joint_range = joint_limits[:, 1] - joint_limits[:, 0]

        goal_joint_pos = random_state.normal(
            loc=self.goal_joint_pos, scale=0.1 * joint_range
        )
        goal_joint_pos = np.clip(goal_joint_pos, joint_limits[:, 0], joint_limits[:, 1])

        # replace state to ensure reachability with current model
        self.goal_simulation.set_qpos("robot0:hand_joint_angles", goal_joint_pos)
        self.goal_simulation.forward()

        # take a few steps to avoid goals that are impossible due to contacts
        for steps in range(2):
            self.goal_simulation.shadow_hand.set_position_control(
                self.goal_simulation.shadow_hand.denormalize_position_control(
                    self.goal_simulation.shadow_hand.zero_control(),
                    relative_action=True,
                )
            )

            self.goal_simulation.step()

        self.goal_joint_pos = (
            self.goal_simulation.shadow_hand.observe().joint_positions()
        )

        return {
            "fingertip_pos": self._get_fingertip_position(self.goal_simulation),
        }

    def current_state(self) -> dict:
        """ Extract current cube goal state """
        return {"fingertip_pos": self._get_fingertip_position(self.mujoco_simulation)}

    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        return {
            "fingertip_pos": goal_state["fingertip_pos"]
            - current_state["fingertip_pos"]
        }

    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        relative_goal = self.relative_goal(goal_state, current_state)
        return {"fingertip_pos": np.linalg.norm(relative_goal["fingertip_pos"])}

    @staticmethod
    def _get_fingertip_position(simulation: ReachSimulation):
        """
        Get absolute fingertip positions in mujoco frame.
        """
        fingertip_pos = np.array(
            [
                simulation.mj_sim.data.get_site_xpos(f"robot0:{site}")
                for site in FINGERTIP_SITE_NAMES
            ]
        )

        fingertip_pos = fingertip_pos.flatten()

        return fingertip_pos
