#!/usr/bin/env python3

import os

import click

from robogym.utils.env_utils import load_env
from robogym.utils.parse_arguments import parse_arguments
from robogym.viewer.holdout_creation_viewer import HoldoutCreationViewer


@click.command()
@click.argument("argv", nargs=-1, required=True)
def main(argv):
    """
    Create holdout env using given config.
    This script is only supported for rearrange environment.

    Example usage:

    python scripts/create_holdout.py envs/rearrange/holdouts/configs/sample.jsonnet
    """
    create_holdout(argv)


def create_holdout(argv):
    names, kwargs = parse_arguments(argv)
    config_path = names[0]
    env = load_env(f"{config_path}::make_env", **kwargs)
    name = os.path.splitext(os.path.basename(config_path))[0]
    viewer = HoldoutCreationViewer(env, name)
    viewer.run()


if __name__ == "__main__":
    main()
