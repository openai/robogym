import logging

from robogym.envs.dactyl.goals.face_cube_solver import FaceCubeSolverGoal

logger = logging.getLogger(__name__)


class ReleaseCubeSolverGoal(FaceCubeSolverGoal):
    def face_threshold(self):
        """
        Dynamic face threshold to use a custom success threshold
        that is lower than the typical threshold
        to assess face alignment once the cube has been fully solved
        :return:
        """
        if self.goal_step < len(self.goal_sequence):
            return self.success_threshold["cube_face_angle"]
        return 0.05

    def _get_goal_action(self):
        """
        Get the required action to achieve current goal state.
        """
        # We solve the cube once and stop generating goals.
        if self.goal_step < len(self.goal_sequence):
            goal = self.goal_sequence[self.goal_step]
        else:
            goal = self.goal_sequence[-1]
            self.reached_terminal_state = True
        return goal
