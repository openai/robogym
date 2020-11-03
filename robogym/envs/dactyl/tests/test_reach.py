from robogym.envs.dactyl.reach import make_env


def test_dactyl_reach():
    env = make_env()
    obs = env.reset()
    expected_joints = (
        "robot0:WRJ1",
        "robot0:WRJ0",
        "robot0:FFJ3",
        "robot0:FFJ2",
        "robot0:FFJ1",
        "robot0:FFJ0",
        "robot0:MFJ3",
        "robot0:MFJ2",
        "robot0:MFJ1",
        "robot0:MFJ0",
        "robot0:RFJ3",
        "robot0:RFJ2",
        "robot0:RFJ1",
        "robot0:RFJ0",
        "robot0:LFJ4",
        "robot0:LFJ3",
        "robot0:LFJ2",
        "robot0:LFJ1",
        "robot0:LFJ0",
        "robot0:THJ4",
        "robot0:THJ3",
        "robot0:THJ2",
        "robot0:THJ1",
        "robot0:THJ0",
    )
    assert env.unwrapped.sim.model.joint_names == expected_joints
    for k, ob in obs.items():
        assert ob.shape[0] > 0
