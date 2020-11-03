#!/usr/bin/env python3
import logging

import click

from robogym.envs.rearrange.common.base import RearrangeEnv
from robogym.utils.env_utils import load_env
from robogym.utils.parse_arguments import parse_arguments
from robogym.viewer.env_viewer import EnvViewer
from robogym.viewer.robot_control_viewer import RobotControlViewer

logger = logging.getLogger(__name__)


@click.command()
@click.argument("argv", nargs=-1, required=False)
@click.option(
    "--teleoperate",
    is_flag=True,
    help="If true, loads environment in teleop mode. Teleop mode is only supported for rearrange environments with TCP robot control modes",
)
def main(argv, teleoperate):
    """
    examine.py is used to display environments.
    \b
    Example uses:
        ./examine.py dactyl/full_perpendicular.py      : displays environment full_perpendicular from envs/dactyl/full_perpendicular.py
        ./examine.py rearrange/blocks.py --teleoperate : loads BlocksRearrangeEnv in teleoperate mode such that the robot can be teleoperated via the keyboard.
    """

    names, kwargs = parse_arguments(argv)

    assert len(names) == 1, "Expected a single argument for the environment."
    env_name = names[0]
    env, args_remaining = load_env(env_name, return_args_remaining=True, **kwargs)

    assert env is not None, print(
        '"{}" doesn\'t seem to be a valid environment'.format(env_name)
    )
    if teleoperate:
        teleop_compatible = (
            isinstance(env.unwrapped, RearrangeEnv)
            and env.parameters.robot_control_params.is_tcp_controlled()
        )
        assert (
            teleop_compatible
        ), "Teleoperation is only supported for rearrange environments with TCP control modes."
    viewer = RobotControlViewer if teleoperate else EnvViewer
    viewer(env).run()


if __name__ == "__main__":
    # This ensures that we spawn new processes instead of forking. This is necessary
    # rendering has to be in spawned process. GUI cannot be in a forked process.
    # set_start_method('spawn', force=True)
    logging.getLogger("").handlers = []
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    main()
