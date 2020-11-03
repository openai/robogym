from os.path import abspath, dirname, join

# This is the absolute path to the root directory for the robogym repo.
ROBOGYM_ROOT_PATH = abspath(join(dirname(__file__), ".."))


def robogym_path(*args):
    """
    Returns an absolute path from a path relative to the robogym repository root directory.
    """
    return join(ROBOGYM_ROOT_PATH, *args)


def pretty(vec, precision=3):
    """
    Returns short, pretty version of a float vector.
    """
    if vec is None or vec.shape[0] == 0:
        return ""
    ret = "["
    max_entries = 6
    for idx in range(vec.shape[0]):
        if idx < 6 or idx > vec.shape[0] - max_entries:
            if vec[idx] >= 0:
                ret += " "
            ret += ("%." + str(precision) + "f ") % vec[idx]
        elif idx == max_entries:
            ret += "... "
    return ret[:-1] + "]"
