from robogym.wrappers import randomizations


class RandomizedFaceDampingWrapper(randomizations.RandomizedDampingWrapper):
    def __init__(self, env=None, damping_range=[1 / 3.0, 3.0], object_name="cube"):
        joint_names = [
            object_name + ":" + name for name in env.unwrapped.face_joint_names
        ]
        super().__init__(env, damping_range, joint_names)
