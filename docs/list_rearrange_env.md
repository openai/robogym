# List of Rearrange Environments
This document lists many instances of rearrange environments where some of them are designed for
 training and some of them are designed for testing generalization of a learned policy. 

## Training Environments

Training environments are classified based on the set of objects and the goal distribution.
By default, each training environments are designed to generate a fixed number of objects.
If you want to train a policy with a variable number of objects, `num_objects` should be
 randomized using the [randomization](env_param_interface.md) interface.

We recommend to inherit the following default configs for all training environments.
```jsonnet
constants: {
    # For the reward, we do the following:
    #  - for every object that gets placed correctly, immediately provide +1
    #  - once all objects are properly placed, the policy needs to hold the goal
    #    state for 2 seconds (in simulation time) until it gets `success_reward` reward
    #  - the next goal is generated
    success_reward: 5.0,
    success_pause_range_s: [0.0, 0.5],
    max_timesteps_per_goal_per_obj: 600,
    vision: True,  # use False if you don't want to use vision observations
    vision_args: {
        image_size: 200,
        camera_names: ['vision_cam_front'],
        mobile_camera_names: ['vision_cam_wrist']
    },
    goal_args: {
        rot_dist_type: 'full',
        randomize_goal_rot: true,
        p_goal_hide_robot:: 1.0,
    },
    # 'obj_pos': l2 distance in meter; 'obj_rot' eular angle distance.
    success_threshold: {'obj_pos': 0.04, 'obj_rot': 0.2},
},
parameters: {
    simulation_params: {
        num_objects: 1,
        max_num_objects: 32,
        object_size: 0.0254,
        used_table_portion: 1.0,
        goal_distance_ratio: 1.0,
        cast_shadows:: false,
        penalty: {
            # Penalty for collisions (wrist camera, table)
            wrist_collision:: 0.0,
            table_collision:: 0.0,
            
            # Penalty for safety stops
            safety_stop_penalty:: 0.0,
        }
    }
}
```

The table below describes a set of training environment.

|Name|File|Config (overwrite)|Description|
|----------|:-------------|:-------------|:-------------|
|blocks reach|blocks_reach.py|-|Place end-of-effector of a robot to the target position. This training environment is not compatible with holdout environments.|
|blocks|blocks_train.py|-| Pushing blocks to targets on the surface of a table|
|blocks (push + pick-and-place)|blocks_train.py|`constants.goal_args.pickup_proba: 0.4`| Pushing or pick-and-placing blocks to targets on the surface of a table or in the air|
|k-composer|composer.py|`parameters.simulation_params.num_max_geoms: k`| Pushing objects to targets on the surface of a table. Each objects are created by randomly composing `[1, k]` meshes.|
|ycb|ycb.py|`parameters.simulation_params.mesh_scale: 0.6`| Pushing ycb objects to targets on the surface of a table|
|mixture|mixture.py|`constants: {normalize_mesh: True, normalized_mesh_size: 0.05}`<br>`parameters.simulation_params.mesh_scale: 1.0`| Pushing objects to targets on the surface of a table, objects are randomly sample from ycb or simple geom shapes.|


## Holdout Environments

We design a set of holdout environment to evaluate generalization performance of a learned policy.
To make holdout environment compatible with the recommended training environment, we
 recommend inheriting the same default configs from the training environment.
 
The table below describes a list of holdout environments and their configs.
As a common config for all holdout environments, `n`-objects holdout environment overwrites
 `parameters.simulation_params.num_objects: n`.
Configs for some holdout environment can be found in `robogym/envs/rearrange/holdouts/configs/`.
In this case the table will provide a pointer to these configs.

|Name|File|Config (overwrite)|<div style="width:300px">Description</div> |
|-------|-------|-------|:-------------|
|n-blocks push|blocks.py|`constants.goal_args.rot_dist_type: 'mod90'`|push blocks to targets on the surface of the table.|
|n-blocks flip|blocks.py|`constants.goal_args.rot_randomize_type: 'block'`<br>`parameters.simulation_params.block_appearance: 'openai'`|push and flip blocks to targets on the surface of the table. Target block orientation may require flipping the block.|
|n-blocks duplicate|blocks_duplicate.py|`constants.goal_args.rot_dist_type: 'mod90'`|push `n` identical looking blocks to targets on the surface of the table.|
|n-blocks pick-and-place|blocks_pickandplace.py|`constants.goal_args.rot_dist_type: 'mod90'`|similar to block push, but one object target is in the air.|
|n-blocks stack (`1 < n < 5`)|blocks_stack.py|`constants.goal_args.rot_dist_type: 'mod90'`|stack `n` blocks into a tower.|
|attached blocks|blocks_attached.py|`constants.goal_args.rot_dist_type: 'mod90'`|rearrange 8 blocks to a particular structured shape (at least one face of each blocks are attached to another).|
|n-ycb push|ycb.py|-|push ycb objects to targets on the surface of the table.|
|n-ycb pick-and-place|ycb_pickandplace.py|-|similar to ycb push, but one object target is in the air.|
|ball_capture|ball_capture.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|ball_in_mug|bin_packing.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|bin_packing|bin_packing.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|bin_unpacking|bin_unpacking.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|bookshelf|bookshelf.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|chess4|chessboard.py|`parameters.simulation_params.num_objects: 4`|see [this figure](assets/all_holdouts.png)|
|chessboard|chess.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|domino5|dominos.py|`parameters.simulation_params.num_objects: 5`<br>`constants.is_holdout: True`|see [this figure](assets/all_holdouts.png)|
|dominoes|dominoes.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|jenga_cross|jenga/cross.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|jenga_tower|jenga/leaning_tower.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|jenga_tower_disassemble|jenga/leaning_tower_disassemble.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|jenga_tower_stack6|jenga/stack6.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|lego_easy_stack2|lego/easy_stack2.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|lego_easy_stack3|lego/easy_stack3.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|lego_stack2|lego/stack2.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|lego_stack2L|lego/stack2L.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|lego_stack3|lego/stack3.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|lego_stack5|lego/stack5.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow|rainbow.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow_balance|rainbow_build/rainbow_balance.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow_stack2|rainbow_build/rainbow_stack2.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow_stack4|rainbow_build/rainbow_stack4.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow_stack6|rainbow_build/rainbow_stack6.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow_stack_inv2|rainbow_build/rainbow_stack_inv2.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow_stack_inv4|rainbow_build/rainbow_stack_inv4.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|rainbow_stack_inv6|rainbow_build/rainbow_stack_inv6.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|table_setting|table_setting.py|`constants.goal_args.randomize_goal_rot: False`<br>`constants.success_needed: 1`<br>`parameters.simulation_params.num_objects: 5`|see [this figure](assets/all_holdouts.png)|
|tangram|tangram.jsonnet|-|see [this figure](assets/all_holdouts.png)|
|wordblocks|wordblocks.py|`constants.goal_args.randomize_goal_rot: False`<br>`constants.success_needed: 1`<br>`parameters.simulation_params.num_objects: 6`|see [this figure](assets/all_holdouts.png)|
