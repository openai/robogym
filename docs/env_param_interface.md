# Interface for Environment Randomization

Robogym provides a way to intervene the environment parameters during training to support domain randomization and curriculum learning.
This interface is called `randomization`.
`randomization` is used to randomize various aspects of an environment
such as initial state distribution, goal distribution, and transition dynamics.

This document describes an example of using `randomization` to modify the number of objects
that are sampled in a blocks_train environment.

Let's define a blocks_train environment with the default `num_objects: 5`.
```python
from robogym.envs.rearrange.blocks_train import make_env

env = make_env(
    parameters={
        'simulation_params': {
            'num_objects': 5,
            'max_num_objects': 8,
        }
    }
)
```

By setting `num_objects: 5`, and `max_num_objects: 8`, this environment will sample 5 blocks on
by default, while allowing to use the range `[1, 8]` for `num_objects`.

One can check how many objects are sampled by looking at `obj_pos` observation:
```python
obs = env.reset()

print(obs['obj_pos'])
# example output: 
# [[1.70972328, 0.42136078, 0.51167315],
#  [1.11886926, 0.53096005, 0.51168124],
#  [1.33472491, 0.93510741, 0.51167315],
#  [1.17595694, 0.78276944, 0.51167315],
#  [1.47985358, 0.49202608, 0.51167315],
#  [0.        , 0.        , 0.        ],
#  [0.        , 0.        , 0.        ],
#  [0.        , 0.        , 0.        ]]
```
Note that the first 5 indices are filled with `(x, y, z)` position of 5 objects, and the
remaining indices are filled with a placeholder.

`randomization` interface allows one to modify the environment parameter (`num_objects` in
 this case) whenever environment reset.

`num_objects` parameter can be accessed and modified as follows
```python
param = env.unwrapped.randomization.get_parameter("parameters:num_objects")

print(param.get_value())
# example output:
# 5

# Set num_object: 3 for the next episode
param.set_value(3)

obs = env.reset()
print(obs['obj_pos'])
# example output:
# [[1.33433012 0.86198878 0.51169309]
#  [1.4010548  0.70733166 0.51169309]
#  [1.25599204 0.93231204 0.51167315]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.        ]]

print(param.get_value())
# example output:
# 3
```
This example shows that the environment uses a newly set environment parameter for the following
 episode.

`"parameters:num_objects"` is a unique name to access a particular environment parameter, and
 there are many more parameters that one can use to randomize environment.
A few example names of parameters include the followings:
* `parameters:goal_distance_ratio`: parameter for defining curriculum on the distance from
 objects' initial position to target positions.
* `parameters:goal_rot_weight`: parameter for defining curriculum on the threshold for measuring
 success for object orientation goal.
* `sim.gravity:value`: parameter for randomizing the gravity of the environment.
* `sim.dof_damping_robot:mean`: mean of dof_damping randomization distribution for robot
* `sim.dof_damping_robot:std`: std of dof_damping randomization distribution for robot
* ...

There are many more parameters implemented for robogym environment.
Check the [RobotEnv.build_randomization](../robogym/robot_env.py#L1041) function for more details.
