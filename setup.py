#!/usr/bin/env python3
from setuptools import find_packages, setup


def setup_robogym():
    setup(
        name="robogym",
        version=open("ROBOGYM_VERSION").read(),
        packages=find_packages(),
        install_requires=[
            # Fixed versions
            "click==7.0",
            "collision==1.2.2",
            "gym==0.15.3",
            "kociemba==1.2.1",
            "mujoco-py==2.0.2.13",
            "pycuber==0.2.2",
            "matplotlib==3.1.2",
            "transforms3d==0.3.1",
            # Minimum versions
            "jsonnet>=0.14.0",
            "pytest>=4.6.9",
            "scikit-learn>=0.21.3",
            "trimesh>=3.5.23",
            "mock>=4.0.2",
        ],
        python_requires=">=3.7.4",
        description="OpenAI Robogym Robotics Environments",
        include_package_data=True,
    )


setup_robogym()
