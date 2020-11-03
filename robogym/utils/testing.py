import numpy as np


def assert_dict_match(d1: dict, d2: dict, eps: float = 1e-6):
    """Assert if two dictionary variables are different.

    :param eps: the threshold used when comparing two float values from dicts.
    """
    assert sorted(d1.keys()) == sorted(d2.keys())
    for k in d1:
        assert isinstance(d1[k], type(d2[k]))  # same type
        if isinstance(d1[k], np.ndarray):
            assert np.allclose(d1[k], d2[k], atol=eps)
        elif isinstance(d1[k], (float, np.float32, np.float64)):
            assert abs(d1[k] - d2[k]) < eps, f"{k}: {d1[k]} != {d2[k]}"
        elif isinstance(d1[k], dict):
            assert_dict_match(d1[k], d2[k])
        else:
            assert d1[k] == d2[k], f"{k}: {d1[k]} != {d2[k]}"
