import itertools
import typing

import mujoco_py
import mujoco_py.generated.const


def joint_qpos_ids(model, joint_name: str) -> typing.List[int]:
    addr = model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def joint_qpos_ids_from_prefix(model, joint_prefix):
    qpos_ids_list = [
        joint_qpos_ids(model, name)
        for name in model.joint_names
        if name.startswith(joint_prefix)
    ]
    return list(itertools.chain.from_iterable(qpos_ids_list))


def joint_qvel_ids(model, joint_name: str) -> typing.List[int]:
    addr = model.get_joint_qvel_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def joint_qvel_ids_from_prefix(model, joint_prefix):
    qvel_ids_list = [
        joint_qvel_ids(model, name)
        for name in model.joint_names
        if name.startswith(joint_prefix)
    ]
    return list(itertools.chain.from_iterable(qvel_ids_list))


def joint_type_name(joint_type: int) -> str:
    if joint_type == mujoco_py.generated.const.JNT_FREE:
        return "free"
    if joint_type == mujoco_py.generated.const.JNT_BALL:
        return "ball"
    if joint_type == mujoco_py.generated.const.JNT_SLIDE:
        return "slide"
    if joint_type == mujoco_py.generated.const.JNT_HINGE:
        return "hinge"

    raise AssertionError(f"unsupported joint type: {joint_type}")
