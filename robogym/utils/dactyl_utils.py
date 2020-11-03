# This function can'be removed yet. There are two places that still need it: DactylReachEnv and
# RandomizedJointLimitWrapper. The latter can't be changed until the old environments are refactored. And the first
# one relies on it for initialization. An additional refactor is needed to remove this util.
def actuated_joint_range(sim):
    joint_limits = sim.model.jnt_range.copy()
    for a_idx, name in enumerate(sim.model.actuator_names):
        j_idx = sim.model.joint_names.index(name.replace("A_", ""))
        actuated_limits = sim.model.actuator_ctrlrange[a_idx, :]
        joint_limits[j_idx, 0] = max(joint_limits[j_idx, 0], actuated_limits[0])
        joint_limits[j_idx, 1] = min(joint_limits[j_idx, 1], actuated_limits[1])
        # avoid cases where limits cross
        joint_limits[j_idx, 1] = max(joint_limits[j_idx, 0], joint_limits[j_idx, 1])

    return joint_limits
