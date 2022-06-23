#!/usr/bin/env python3
from setuptools import find_packages, setup
from distutils.extension import Extension
from os.path import dirname, realpath, join
import platform


SRC_DIR = dirname(realpath(__file__))


class CallableList(object):
    def __init__(self):
        self.index = 0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        print("calling getitem!")
        return None

    def __iter__(self):
        return self
        
    def __next__(self):
        if self.index == 1:
            raise StopIteration()
        if self.index == 0:
            import mujoco
            self.index += 1
            return join(mujoco.__path__[0], "include")


def setup_robogym():
    if platform.system() == "Windows":
        extra_compile_args = ['/O2', '-std:c++17']
    else:
        extra_compile_args = ['-O3', '-std=c++17', '-Wno-#warnings', '-Wno-cpp', '-Wno-unused-function', '-Wno-deprecated-declarations']
    setup(
        name="robogym",
        version=open("ROBOGYM_VERSION").read(),
        packages=find_packages(),
        install_requires=[
            # Fixed versions
            "click",
            "collision==1.2.2",
            "gym",
            "kociemba==1.2.1",
            "mujoco>=2.2.0",
            "pycuber==0.2.2",
            "matplotlib",
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
        ext_modules=[
            Extension('robogym.mujoco.callbacks',
                    sources=[join(SRC_DIR, "robogym", "mujoco", "callbacks.cpp")],
                    extra_compile_args=extra_compile_args,
                    include_dirs=CallableList(),
                    language='c++')
        ]
    )


setup_robogym()
