import logging

from gym.wrappers import *  # noqa # type: ignore

from .cube import *  # noqa # type: ignore
from .dactyl import *  # noqa # type: ignore
from .face import *  # noqa # type: ignore
from .parametric import *  # noqa # type: ignore
from .randomizations import *  # noqa # type: ignore
from .util import *  # noqa # type: ignore

logger = logging.getLogger(__name__)


def apply_named_wrappers(env, wrappers):
    # lazy init to avoid import loop
    # import all wrappers so that they can be referred to without qualification

    for wrapper in wrappers:
        wrapper_args = {} if (len(wrapper) == 1 or wrapper[1] is None) else wrapper[1]
        logger.info("Adding Wrapper %s with args %s" % (wrapper[0], wrapper_args))
        env = eval(wrapper[0])(env, **wrapper_args)

    return env


def edit_wrappers(*, wrappers, insert_above=[], insert_below=[], replace=[], delete=[]):
    """
        Edit list of wrappers with 4 operations. Order of operations is insert_above, insert_below, replace, delete.
        Args:
        - insert_above (list): list of lists, where each item contains wrapper name where we are inserting above and the wrapper to insert.
                e.g. insert_above=[["RandomizedCubeSizeWrapper", ["RandomizedTimestepWrapper", wrapper_args_dict]], ...]
        - insert_below (list): same as insert_above except inserts below
        - replace (list): same as insert_above syntax except replaces the target wrapper
        - delete (list): list of wrapper names. e.g. delete=["RandomizedCubeSizeWrapper", "RandomizedTimestepWrapper"] to turn of those two wrappers
        Returns: list of wrappers
    """
    # Insert Above
    for _insert_above in insert_above:
        try:
            ind = [wrapper[0] for wrapper in wrappers].index(_insert_above[0])
            wrappers.insert(ind, _insert_above[1])
        except ValueError:
            logger.warning(_insert_above[0] + " not found in wrappers!!!")
            assert False

    # Insert Below
    for _insert_below in insert_below:
        try:
            ind = [wrapper[0] for wrapper in wrappers].index(_insert_below[0]) + 1
            wrappers.insert(ind, _insert_below[1])
        except ValueError:
            logger.warning(_insert_below[0] + " not found in wrappers!!!")
            assert False

    # Replace
    for _replace in replace:
        try:
            ind = [wrapper[0] for wrapper in wrappers].index(_replace[0])
            wrappers[ind] = _replace[1]
        except ValueError:
            logger.warning(_replace[0] + " not found in wrappers!!!")
            assert False

    # Delete
    for _delete in delete:
        try:
            ind = [wrapper[0] for wrapper in wrappers].index(_delete)
            wrappers.pop(ind)
        except ValueError:
            logger.warning(_delete + " not found in wrappers!!!")
            assert False

    return wrappers


def find_wrapper(env_top, search_string):
    """
    recursively search for env wrapper containing the given string

    :param env_top: top-level environment
    :param search_string: (string) string to find in wrapper class name
    :return: environment, (optional) stack of searched environments
    """
    stack = []
    curr_env = env_top
    assert curr_env is not env_top.unwrapped
    while search_string not in curr_env.class_name():
        stack.append(curr_env)
        curr_env = curr_env.env
        assert curr_env is not env_top.unwrapped
    assert search_string in curr_env.class_name()
    return curr_env, stack
