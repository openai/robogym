import logging

from robogym.wrappers.named_wrappers import apply_named_wrappers, edit_wrappers

logger = logging.getLogger(__name__)


def construct_default_wrappers(
    *,
    randomize: bool,
    n_action_bins: int,
    fixed_wrist: bool,
    adr_wrapper,
    relative_goal_wrapper: bool = False,
    drop_reward: float = -20.0,
    default_wrappers=None,
    min_episode_length: int = -1
):
    """
    Construct default list of wrappers.

    Args:
    - randomize (bool): use randomizations. See default-wrapper-base.jsonnet for default
        randomization blocks
    - vision_args (dict): see base-vision.jsonnet for examples
    - n_action_bins (int or None=DiscretizeActionWrapper.DEFAULT_BINS): number of discrete bins
    - min_episode_length: If positive, a dropped cube at a timestep below min_epsiode_length will
        not trigger a 'done'. A penalty for cube dropping is only returned on the first frame.
    Returns: list of wrappers
    """
    wrappers = []

    # actions should be clipped immediately before sending to env
    if fixed_wrist:
        wrappers.append(["FixedWristWrapper"])  # must be inside clipping
    wrappers.append(["ClipActionWrapper"])

    wrappers.append(
        [
            "StopOnFallWrapper",
            dict(min_episode_length=min_episode_length, drop_reward=drop_reward,),
        ]
    )

    if randomize:
        wrappers.append(["BacklashWrapper"])

        if adr_wrapper is not None:
            wrappers.append(adr_wrapper)

        wrappers += default_wrappers["pre_obsnoise_randomizations"]
        noise_levels = default_wrappers["default_observation_noise_levels"]
        observation_delay_levels = default_wrappers["default_observation_delay_levels"]
    else:
        noise_levels = default_wrappers["default_no_noise_levels"]
        observation_delay_levels = default_wrappers[
            "default_no_observation_delay_levels"
        ]

    wrappers.append(["ObservationDelayWrapper", dict(levels=observation_delay_levels)])

    wrappers.append(
        ["RandomizeObservationWrapper", dict(levels=noise_levels)]
    )  # must happen before angle observation wrapper

    wrappers.append(
        ["SmoothActionWrapper"]
    )  # it is important that this gets applied before noise is added

    if relative_goal_wrapper:
        wrappers.append(["RelativeGoalWrapper", dict(obs_prefix="cube_")])

    if randomize:
        wrappers += default_wrappers["post_obsnoise_randomizations"]

    wrappers.append(["AngleObservationWrapper"])
    wrappers.append(
        [
            "UnifiedGoalObservationWrapper",
            dict(goal_parts=["pos", "quat", "face_angle"]),
        ]
    )
    wrappers.append(["ClipObservationWrapper"])
    wrappers.append(["ClipRewardWrapper"])

    wrappers.append(["PreviousActionObservationWrapper"])
    wrappers.append(["RewardObservationWrapper", {"reward_inds": [1, 2]}])

    wrappers.append(["DiscretizeActionWrapper", {"n_action_bins": n_action_bins}])

    return wrappers


def apply_wrappers(
    env,
    randomize,
    wrappers=None,
    n_action_bins=None,
    fixed_wrist=False,
    insert_above=[],
    insert_below=[],
    replace=[],
    delete=[],
    adr_wrapper=None,
    relative_goal_wrapper=False,
    drop_reward=-20.0,
    default_wrappers=None,
    min_episode_length=-1,
):
    if wrappers is None:
        wrappers = construct_default_wrappers(
            randomize=randomize,
            n_action_bins=n_action_bins,
            fixed_wrist=fixed_wrist,
            drop_reward=drop_reward,
            adr_wrapper=adr_wrapper,
            relative_goal_wrapper=relative_goal_wrapper,
            default_wrappers=default_wrappers,
            min_episode_length=min_episode_length,
        )

    wrappers = edit_wrappers(
        wrappers=wrappers,
        insert_above=insert_above,
        insert_below=insert_below,
        replace=replace,
        delete=delete,
    )
    env = apply_named_wrappers(env, wrappers)

    return env


def get_vision_wrapper_args(input_vision_args, cube_type):
    vision_args = (input_vision_args or {}).copy()
    if "vision_env_args" not in vision_args:
        vision_args["vision_env_args"] = {}

    if cube_type == "full-perpendicular":
        vision_env_args = {
            "hide_target": True,
            "cube_appearance": "vision",
        }
    elif cube_type in "face-perpendicular":
        vision_env_args = {
            "hide_target": True,
            "randomize": False,
            "n_random_initial_steps": 0,
        }
    elif cube_type == "locked":
        vision_env_args = {
            "hide_target": True,
            "cube_appearance": "material",
            "randomize": False,
            "n_random_initial_steps": 0,
        }
    else:
        vision_env_args = {}

    vision_args["vision_env_args"].update(vision_env_args)
    return vision_args
