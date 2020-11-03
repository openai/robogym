import functools
import typing

import attr
import numpy as np

from robogym.envs.dactyl.observation.reach import (
    GoalFingertipPosObservation,
    GoalIsAchievedObservation,
)
from robogym.envs.dactyl.observation.shadow_hand import (
    MujocoShadowhandAbsoluteFingertipsObservation,
    MujocoShadowHandJointPosObservation,
    MujocoShadowHandJointVelocityObservation,
)
from robogym.goal.goal_generator import GoalGenerator
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.observation.goal import GoalObservationProvider
from robogym.observation.mujoco import MujocoObservationProvider, ObservationProvider
from robogym.robot.shadow_hand.hand_forward_kinematics import FINGERTIP_SITE_NAMES
from robogym.robot.shadow_hand.mujoco.mujoco_shadow_hand import MuJoCoShadowHand
from robogym.robot_env import ObservationMapValue as omv
from robogym.robot_env import RobotEnv, RobotEnvConstants, RobotEnvParameters
from robogym.wrappers import dactyl, randomizations, util

DEFAULT_NOISE_LEVELS: typing.Dict[str, dict] = {
    "fingertip_pos": {"uncorrelated": 0.001, "additive": 0.001},
}

NO_NOISE_LEVELS: typing.Dict[str, dict] = {
    key: {} for key in DEFAULT_NOISE_LEVELS.keys()
}


@attr.s(auto_attribs=True)
class ReachEnvParameters(RobotEnvParameters):
    """ Parameters of the shadow hand reach env - possible to change for each episode. """

    pass


@attr.s(auto_attribs=True)
class ReachEnvConstants(RobotEnvConstants):
    """ Parameters of the shadow hand reach env - same for all episodes. """

    success_threshold: dict = {"fingertip_pos": 0.025}

    # If specified, freeze all other fingers.
    active_finger: typing.Optional[str] = None

    # Overwrite the following constants regarding rewards.
    successes_needed: int = 50

    max_timesteps_per_goal: int = 150


class ReachSimulation(SimulationInterface):

    """
    Simulation interface for shadow hand reach env.
    """

    # Just a floor
    FLOOR_XML = "floor/basic_floor.xml"
    # Target fingertip sites.
    TARGET_XML = "shadowhand_reach/target.xml"
    # Robot hand xml
    HAND_XML = "robot/shadowhand/main.xml"
    # XML with default light
    LIGHT_XML = "light/default.xml"

    def __init__(self, sim):
        super().__init__(sim)
        self.enable_pid()
        self.shadow_hand = MuJoCoShadowHand(self)

    @classmethod
    def build(cls, n_substeps: int = 10):
        """Construct a ShadowHandReachSimulation object.

        :param n_substeps: (int) sim.nsubsteps, num of substeps
        :return: a ShadowHandReachSimulation object with properly constructed sim.
        """
        xml = MujocoXML()
        xml.add_default_compiler_directive()

        xml.append(
            MujocoXML.parse(cls.FLOOR_XML).set_named_objects_attr(
                "floor", tag="body", pos=[1, 1, 0]
            )
        )

        target = MujocoXML.parse(cls.TARGET_XML)

        colors = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ]

        for site, color in zip(FINGERTIP_SITE_NAMES, colors):
            target.set_named_objects_attr(
                f"target_{site}",
                pos=[0.5, 0.5, 0.0],
                type="sphere",
                rgba=color,
                size=0.005,
            )

        xml.append(target)

        xml.append(
            MujocoXML.parse(cls.HAND_XML)
            .add_name_prefix("robot0:")
            .set_named_objects_attr(
                "robot0:hand_mount",
                tag="body",
                pos=[1.0, 1.25, 0.15],
                euler=[np.pi / 2, 0, np.pi],
            )
            .remove_objects_by_name("robot0:annotation:outer_bound")
            # Remove hand base free joint so that hand is immovable
            .remove_objects_by_name("robot0:hand_base")
        )

        xml.append(MujocoXML.parse(cls.LIGHT_XML))

        simulation = cls(xml.build(nsubsteps=n_substeps))

        # Move fingers out of the way.
        simulation.shadow_hand.set_position_control(
            simulation.shadow_hand.denormalize_position_control(
                simulation.shadow_hand.zero_control()
            )
        )

        for _ in range(20):
            simulation.step()

        return simulation


class ReachEnv(RobotEnv[ReachEnvParameters, ReachEnvConstants, ReachSimulation]):
    """
    Environment with the ShadowHand and a locked cube (i.e. no moving pieces, just a solid
    block).
    """

    def _build_observation_providers(self):
        """
        Initialize observation providers for the environment.
        """
        providers: typing.Dict[str, ObservationProvider] = {
            "mujoco": MujocoObservationProvider(self.mujoco_simulation),
            "goal": GoalObservationProvider(lambda: self.goal_info()),
        }

        return providers

    def _default_observation_map(self):
        return {
            "qpos": omv({"mujoco": MujocoShadowHandJointPosObservation}),
            "qvel": omv({"mujoco": MujocoShadowHandJointVelocityObservation}),
            "fingertip_pos": omv(
                {"mujoco": MujocoShadowhandAbsoluteFingertipsObservation}
            ),
            "goal_fingertip_pos": omv({"goal": GoalFingertipPosObservation}),
            "is_goal_achieved": omv({"goal": GoalIsAchievedObservation}),
        }

    @classmethod
    def build_goal_generation(
        cls, constants, mujoco_simulation: ReachSimulation
    ) -> GoalGenerator:
        """ Construct a goal generation object """
        goal_simulation = ReachSimulation.build(n_substeps=mujoco_simulation.n_substeps)
        sim = goal_simulation.mj_sim

        # Make sure fingers are separated.
        # For transfer, want to make sure post-noise locations are achievable.
        sim.model.geom_margin[:] = sim.model.geom_margin + 0.002

        from robogym.envs.dactyl.goals.shadow_hand_reach_fingertip_pos import (
            FingertipPosGoal,
        )

        return FingertipPosGoal(mujoco_simulation, goal_simulation)

    @classmethod
    def build_simulation(cls, constants, parameters):
        return ReachSimulation.build(n_substeps=constants.mujoco_substeps)

    @classmethod
    def build_robot(cls, mujoco_simulation, physical):
        return mujoco_simulation.shadow_hand

    def _render_callback(self, _sim, _viewer):
        """ Set a render callback """
        goal_fingertip_pos = self._goal["fingertip_pos"].reshape(-1, 3)
        for finger_idx, site in enumerate(FINGERTIP_SITE_NAMES):
            goal_pos = goal_fingertip_pos[finger_idx]
            site_id = _sim.model.site_name2id(f"target_{site}")
            _sim.data.site_xpos[site_id] = goal_pos

    def _reset(self):
        super()._reset()

        self.constants.success_pause_range_s = (0.0, 0.5)

    def apply_wrappers(self, **wrapper_params):
        """
        Apply wrappers to the environment.
        """
        self.constants: ReachEnvConstants

        env = util.ClipActionWrapper(self)

        if self.constants.active_finger is not None:
            env = dactyl.FingerSeparationWrapper(
                env, active_finger=self.constants.active_finger
            )

        if self.constants.randomize:
            env = randomizations.RandomizedActionLatency(env)
            env = randomizations.RandomizedBodyInertiaWrapper(env)
            env = randomizations.RandomizedTimestepWrapper(env)
            env = randomizations.RandomizedRobotFrictionWrapper(env)
            env = randomizations.RandomizedGravityWrapper(env)
            env = dactyl.RandomizedPhasespaceFingersWrapper(env)
            env = dactyl.RandomizedRobotDampingWrapper(env)
            env = dactyl.RandomizedRobotKpWrapper(env)
            noise_levels = DEFAULT_NOISE_LEVELS
        else:
            noise_levels = NO_NOISE_LEVELS

        # must happen before angle observation wrapper
        env = randomizations.RandomizeObservationWrapper(env, levels=noise_levels)

        if self.constants.randomize:
            env = dactyl.FingersFreezingPhasespaceMarkers(env)
            env = randomizations.ActionNoiseWrapper(env)

        env = util.SmoothActionWrapper(
            env
        )  # this get's applied before noise is added (important)
        env = util.RelativeGoalWrapper(env)
        env = util.UnifiedGoalObservationWrapper(env, goal_parts=["fingertip_pos"])
        env = util.ClipObservationWrapper(env)
        env = util.ClipRewardWrapper(env)
        env = util.PreviousActionObservationWrapper(env)
        env = util.DiscretizeActionWrapper(
            env, n_action_bins=self.constants.n_action_bins
        )

        # Note: Recording wrapper is removed here to favor simplicity.
        return env


make_simple_env = functools.partial(ReachEnv.build, apply_wrappers=False)
make_env = ReachEnv.build
