"""
Logistic functions

Function List:
1. logistic_sigmoid(x: float, a: float) -> float: calculate the normalized logistic sigmoid
       with slope parameter a
2. clipped_logistic_sigmoid(x: float, a: float) -> float: calculate the normalized logistic sigmoid
       with slope parameter a clipped to a [0, 1] output range
"""
import numpy as np


def logistic_sigmoid(x: float, a: float) -> float:
    """
    Calculates the normalized logistic sigmoid as a function of x with parameterization on a

    This function will be symmetric about 0.5

    :param x: input value for calculation. Range: -inf to inf
              will not clip at values to 0 and 1 outside of the 0 to 1 input range
    :param a: value of the slope of the sigmoid. Values range from 0.5 for slope ~ 1 to
              1.0 for slope ~ infinity. There's very little signal at a < 0.5
    :return: the value of the normalized logistic sigmoid at x
    """

    # set epsilon to be small. this is so we don't have divide by zero conditions
    epsilon: float = 0.0001
    # clip a to be between (0 + epsilon) and (1 - epsilon)
    min_param_a: float = 0.0 + epsilon
    max_param_a: float = 1.0 - epsilon
    a = np.maximum(min_param_a, np.minimum(max_param_a, a))
    # set a to be asymptotic at 1 and zero at 0
    a = 1 / (1 - a) - 1

    # calculate the numerator and denominator terms for the normalized sigmoid
    A: float = 1.0 / (1.0 + np.exp(0 - ((x - 0.5) * a * 2.0)))
    B: float = 1.0 / (1.0 + np.exp(a))
    C: float = 1.0 / (1.0 + np.exp(0 - a))
    y: float = (A - B) / (C - B)

    return y


def clipped_logistic_sigmoid(x: float, a: float) -> float:
    """
    Calculates the normalized logistic sigmoid as a function of x with parameterization on a

    This function will be symmetric about 0.5

    :param x: input value for calculation range: Range: -inf to inf, effective range 0 to 1
              will output 0 for values below 0 and 1 for values above 1
    :param a: value of the slope of the sigmoid. Values range from 0.5 for slope ~ 1 to
              1.0 for slope ~ infinity. There's very little signal at a < 0.5
    :return: the value of the normalized logistic sigmoid at x
    """

    # clip values below zero and above one
    x = np.maximum(x, 0.0)
    x = np.minimum(x, 1.0)

    # set epsilon to be small. this is so we don't have divide by zero conditions
    epsilon: float = 0.0001
    # clip a to be between (0 + epsilon) and (1 - epsilon)
    min_param_a: float = 0.0 + epsilon
    max_param_a: float = 1.0 - epsilon
    a = np.maximum(min_param_a, np.minimum(max_param_a, a))
    # set a to be asymptotic at 1 and zero at 0
    a = 1 / (1 - a) - 1

    # calculate the numerator and denominator terms for the normalized sigmoid
    A: float = 1.0 / (1.0 + np.exp(0 - ((x - 0.5) * a * 2.0)))
    B: float = 1.0 / (1.0 + np.exp(a))
    C: float = 1.0 / (1.0 + np.exp(0 - a))
    y: float = (A - B) / (C - B)

    return y
