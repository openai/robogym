import glob
import json
import os
from copy import deepcopy
from functools import partial
from runpy import run_path

import _jsonnet
import numpy as np
from gym.spaces import Box, Dict, Tuple


class InvalidSimulationError(Exception):
    pass


def gym_space_from_arrays(arrays):
    """ Define environment observation space using an example observation """
    if isinstance(arrays, np.ndarray):
        ret = Box(-np.inf, np.inf, arrays.shape, np.float32)
        ret.flatten_dim = np.prod(ret.shape)
    elif isinstance(arrays, (tuple, list)):
        ret = Tuple([gym_space_from_arrays(arr) for arr in arrays])
    elif isinstance(arrays, dict):
        ret = Dict(dict([(k, gym_space_from_arrays(v)) for k, v in arrays.items()]))
    else:
        raise TypeError(f"Array is of unsupported type: {type(arrays)}")
    return ret


def merge_dict_recursive(d1, d2):
    ret = deepcopy(d1)

    for k, v in d2.items():
        if k not in d1 or not isinstance(v, dict):
            ret[k] = v
        else:
            ret[k] = merge_dict_recursive(d1[k], v)

    return ret


def get_function(obj):
    if callable(obj):
        # Should only be used in tests!
        return obj

    name = obj["function"]
    extra_args = obj.get("args", {})
    module_path, function_name = name.rsplit(":", 1)
    result = getattr(__import__(module_path, fromlist=(function_name,)), function_name)

    if len(extra_args) > 0:

        def result_wrapper(*args, **kwargs):
            actual_kwargs = merge_dict_recursive(extra_args, kwargs)
            return result(*args, **actual_kwargs)

        return result_wrapper
    else:
        return result


class MakeEnvFinder:

    ENV_PATTERNS = [
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../envs", "**", "*.py")
        )
    ]

    @classmethod
    def find(cls, pattern, fun_name=None):
        if pattern.endswith("py") and os.path.exists(pattern):
            if fun_name is None:
                fun_name = "make_env"

            print(f"Loading environment from {pattern}::{fun_name}")
            module = run_path(pattern)
            make_env = module[fun_name]
            return make_env
        elif pattern.endswith((".jsonnet", ".libsonnet", ".json")) and os.path.exists(
            pattern
        ):

            def resolve_fun_name(data, fun_name):
                for elem in fun_name.split("."):
                    data = data[elem]
                return data

            print(f"Loading environment from {pattern}::{fun_name}")
            if pattern.endswith(".json"):
                with open(pattern, "r") as f:
                    data = json.load(f)
            else:
                data = json.loads(_jsonnet.evaluate_file(pattern))
            if fun_name is not None:
                resolved_data = resolve_fun_name(data, fun_name)
            else:
                # Auto-detect a working fun_name
                candidates = ["make_env", "machine_pools.evaluator.args.make_env"]
                resolved_data = None
                for candidate in candidates:
                    try:
                        resolved_data = resolve_fun_name(data, candidate)
                        break
                    except KeyError:
                        pass
                assert (
                    resolved_data is not None
                ), "could not auto-detect a function name; please provide it (e.g. via `::machine_pools.bch-b1.args.make_env`)"

            make_env = get_function(resolved_data)
            return make_env
        else:
            matching = [
                m for p in cls.ENV_PATTERNS for m in glob.glob(p, recursive=True)
            ]
            matching = [match for match in matching if match.find(pattern) > -1]
            matching = [
                match
                for match in matching
                if not os.path.basename(match).startswith("test_")
            ]
            assert len(matching) < 2, "Found multiple environments matching %s" % str(
                matching
            )
            if len(matching) == 1:
                matching = matching[0]
                if matching.endswith(".py") and fun_name is not None:
                    matching += "::" + fun_name
                return cls.find(matching)
            else:
                assert None


def load_env(
    pattern,
    make_env_finder=MakeEnvFinder,
    arg_filter=None,
    return_args_remaining=False,
    **kwargs,
):
    args_remaining = {}
    pattern = pattern.split("::")
    fun_name = None

    if len(pattern) == 1:
        pattern = pattern[0]
    else:
        assert len(pattern) == 2
        pattern, fun_name = pattern[0], pattern[1]

    make_env = make_env_finder.find(pattern, fun_name=fun_name)
    assert make_env is not None, f"No environment found matching {pattern}::{fun_name}"

    if arg_filter is not None:
        kwargs, args_remaining = arg_filter(make_env, kwargs)

    make_env = partial(make_env, **kwargs)
    env = make_env()

    if return_args_remaining:
        return env, args_remaining
    else:
        return env
