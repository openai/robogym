from enum import Enum

OPT_FIELDS = {
    "apirate",
    "collision",
    "cone",
    "density",
    "disableflags",
    "enableflags",
    "gravity",
    "impedance",
    "impratio",
    "integrator",
    "iterations",
    "jacobian",
    "magnetic",
    "mpr_iterations",
    "mpr_tolerance",
    "noslip_iterations",
    "noslip_tolerance",
    "o_margin",
    "o_solimp",
    "o_solref",
    "reference",
    "solver",
    "timestep",
    "tolerance",
    "uintptr",
    "viscosity",
    "wind",
}


"""
follow mujoco-py order:

cdef enum USER_DEFINED_ACTUATOR_PARAMS:
    IDX_PROPORTIONAL_GAIN = 0,
    IDX_INTEGRAL_TIME_CONSTANT = 1,
    IDX_INTEGRAL_MAX_CLAMP = 2,
    IDX_DERIVATIVE_TIME_CONSTANT = 3,
    IDX_DERIVATIVE_GAIN_SMOOTHING = 4,
    IDX_ERROR_DEADBAND = 5,

"""
PID_GAIN_PARAMS = [
    "pid_kp",
    "pid_ti",
    "pid_imax_clamp",
    "pid_td",
    "pid_dsmooth",
    "pid_error_deadband",
]


class MujocoEquality(Enum):
    mjEQ_CONNECT = 0  # connect two bodies at a point (ball joint)
    mjEQ_WELD = 1  # fix relative position and orientation of two bodies
    mjEQ_JOINT = 2  # couple the values of two scalar joints with cubic
    mjEQ_TENDON = 3  # couple the lengths of two tendons with cubic
    mjEQ_DISTANCE = 4  # fix the contact distance betweent two geoms
