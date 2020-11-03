import logging
import typing

import numpy as np
import pycuber

from robogym.envs.dactyl.common import cube_utils
from robogym.goal.goal_generator import GoalGenerator
from robogym.utils import rotation
from robogym.utils.rubik_utils import solve_fast

logger = logging.getLogger(__name__)


class GoalAction(typing.NamedTuple):
    face_idx: int
    face_angle: float
    pycuber_action: str
    num_remaining_actions: int


class RubikCubeSolver(GoalGenerator):
    """
    Solve the cube using the kociemba rubik cube solver
    """

    # Mapping from Mujoco face id to pycuber action.
    FACE_ACTIONS = tuple("LRFBDU")

    # Mapping from Mujoco face id to cube color.
    FACE_COLORS = tuple("ROGBWY")

    # Mapping from Mujoco face id to rotation sign. Note that pycuber
    # rotation angle needs to be flipped for certain faces.
    FACE_SIGNS = [1, -1, 1, -1, 1, -1]

    def __init__(
        self,
        mujoco_simulation,
        face_geom_names: typing.List[str],
        num_scramble_steps: int,
    ):
        """
        Create new FaceCubeSolverGoalGenerator object

        :param mujoco_simulation: A SimulationInterface object for a mujoco simulation considered
        :param success_threshold: Dictionary of threshold levels for cube orientation and face
            rotation, for which we consider the cube "aligned" with the goal
        """
        super().__init__()

        assert len(face_geom_names) == 6, "Only full cube can be solved"

        self.mujoco_simulation = mujoco_simulation
        self.face_geom_names = face_geom_names
        self.num_scramble_steps = num_scramble_steps

        self.goal_quat_for_face = cube_utils.face_up_quats(
            mujoco_simulation.sim, "cube:cube:rot", self.face_geom_names
        )

        self._reset_goal_state(pycuber.Cube())

    def _generate_solution_sequence(self, cube):
        """
        Returns an action sequence to solve the cube.
        """

        # Try to find a reasonable length solution.
        step = 0
        solution = None
        while step < 5:
            try:
                max_depth = self.num_scramble_steps + 2 ** step - 1
                solution = solve_fast(cube, max_depth=max_depth)
                break
            except ValueError:
                logging.info(f"Cannot solve cube within {max_depth} steps.")

            step += 1

        assert solution is not None, f"Could not find solution in {max_depth} steps"

        return self._normalize_actions(solution.split())

    @classmethod
    def _normalize_actions(cls, actions):
        """
        Normalize Singmaster Notation to tuple of action in'UDLRFB' and angle in (pi/2, -pi/2).
        """
        normalized_actions = []
        for i, action in enumerate(actions):
            # We track to number of remaining pycuber actions so we can use it to generate
            # optimal solution sequence during goal regeneration.
            num_actions = 1
            num_remaining_actions = len(actions) - i

            if action.endswith("2"):
                num_actions = 2
                action = action[:-1]

            if action.endswith("'"):
                angle = -np.pi / 2
            else:
                angle = np.pi / 2

            face_idx = cls.FACE_ACTIONS.index(action[0])
            angle *= cls.FACE_SIGNS[face_idx]
            normalized_actions.extend(
                [
                    GoalAction(
                        face_idx=face_idx,
                        face_angle=angle,
                        pycuber_action=action,
                        num_remaining_actions=num_remaining_actions,
                    )
                ]
                * num_actions
            )

        return normalized_actions

    def _step_goal(self):
        """
        Move one step forward in the goal sequence. Also update current goal state.
        """
        self.goal_step += 1

        goal_action = self._get_goal_action()
        self.goal_face_state[goal_action.face_idx] += goal_action.face_angle

    def _get_goal_action(self):
        """
        Get the required action to achieve current goal state.
        """
        # We solve the cube first and then reverse this sequence, then solve again, and
        # so on until we run out of time.
        solve_forward = (self.goal_step // len(self.goal_sequence)) % 2 == 0
        if solve_forward:
            goal_idx = self.goal_step % len(self.goal_sequence)
            goal = self.goal_sequence[goal_idx]
        else:
            goal_idx = (
                len(self.goal_sequence) - 1 - (self.goal_step % len(self.goal_sequence))
            )
            goal = self.goal_sequence[goal_idx]
            # flip direction on face rotation
            goal = GoalAction(
                face_idx=goal.face_idx,
                face_angle=-goal.face_angle,
                pycuber_action=goal.pycuber_action,
                num_remaining_actions=goal.num_remaining_actions,
            )

        return goal

    def current_state(self):
        """ Extract current cube goal state """
        cube_pos = self.mujoco_simulation.get_qpos("cube_position")

        return {
            "cube_pos": cube_pos,
            "cube_quat": self.mujoco_simulation.get_qpos("cube_rotation"),
            "cube_face_angle": self.mujoco_simulation.get_face_angles("cube"),
        }

    def reset(self, random_state):
        """ Reset state of the goal generator. """
        cube = self.mujoco_simulation.cube_model.to_pycuber()
        self._reset_goal_state(cube)

    def _reset_goal_state(self, cube):
        self.reached_terminal_state = False
        initial_goal_state = self.current_state()["cube_face_angle"]
        initial_goal_state = rotation.round_to_straight_angles(initial_goal_state)
        logger.info("Reset goal generation state with pycuber state")
        logger.info(cube)

        self.mujoco_simulation.clone_target_from_cube()
        self.mujoco_simulation.align_target_faces()

        self.goal_sequence = self._generate_solution_sequence(cube)

        logger.info("Goal Sequence:")
        self._print_goal_sequence()

        self.goal_face_state = initial_goal_state
        self.goal_step = -1
        self._step_goal()

    def _print_goal_sequence(self):
        sequence = " ".join(a.pycuber_action for a in self.goal_sequence)
        logging.info(f"Solution Sequence: {sequence}")

    def goal_types(self) -> typing.Set[str]:
        return {"rotation", "flip"}
