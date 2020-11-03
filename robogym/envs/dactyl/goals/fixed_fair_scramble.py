import logging

from robogym.envs.dactyl.goals.face_cube_solver import FaceCubeSolverGoal

logger = logging.getLogger(__name__)


class FixedFairScrambleGoal(FaceCubeSolverGoal):
    """
    Generates a series of goals to apply a "fair scramble" to a fully solved Rubik's cube.
    The fair scramble was generated using the WCA app and was not cherry-picked:
        https://www.worldcubeassociation.org/regulations/scrambles/
    Goals are generated in a way to always rotate the top face.
    """

    def _generate_solution_sequence(self, cube):
        solution = "L2 U2 R2 B D2 B2 D2 L2 F' D' R B F L U' F D' L2"
        return self._normalize_actions(solution.split())
