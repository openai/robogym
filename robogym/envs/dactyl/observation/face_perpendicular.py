import numpy as np

from robogym.observation.goal import GoalObservation
from robogym.observation.mujoco import MujocoObservation

MYPY = False

if MYPY:
    from robogym.envs.dactyl.face_perpendicular import FacePerpendicularSimulation

    BaseObservationType = MujocoObservation[FacePerpendicularSimulation]
else:
    BaseObservationType = MujocoObservation


class MujocoFaceAngleObservation(BaseObservationType):
    """
    Implement mujoco base cube face angles.
    """

    def get(self) -> np.ndarray:
        """
        Get cube position.
        """
        return self.provider.mujoco_simulation.get_face_angles("cube")


class GoalCubePosObservation(GoalObservation):
    """
    Implement goal cube position observation.
    """

    def get(self) -> np.ndarray:
        """
        Get cube position.
        """
        assert self.provider.goal
        return self.provider.goal["cube_pos"]


class GoalFaceAngleObservation(GoalObservation):
    """
    Implement goal cube face angle observation.
    """

    def get(self) -> np.ndarray:
        """
        Get cube position.
        """
        assert self.provider.goal
        return self.provider.goal["cube_face_angle"]
