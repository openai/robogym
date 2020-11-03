{
    # Fill this with path to initial state file after
    # running bin/create_holdout.py
    initial_state_path:: null,

    # Fill this with paths to goal state file after
    # running bin/create_holdout.py
    goal_state_paths:: [],

    # changes default mujoco simulation settings
    # to make physics more stable
    mujoco_timestep:: 0.002,
    mujoco_substeps:: std.floor(0.04 / $.mujoco_timestep),

    customized_success_threshold:: null,
    
    # Most holdout envs use fixed goal so 1 success makes most sense.
    # Override this value if your env need more successes.
    successes_needed:: 1,

    # Fill in object configs.
    task_object_configs:: [],

    assert std.length($.task_object_configs) > 0: "Number of object should be > 0",

    # Fill in object configs.
    scene_object_configs:: [],

    # shared mujoco xml among objects.
    shared_settings:: '',

    make_env: {
        "function": "robogym.envs.rearrange.holdout:make_env",
        args: {
            constants: {
                initial_state_path: $.initial_state_path,

                successes_needed: $.successes_needed,

                mujoco_timestep: $.mujoco_timestep,

                [if $.customized_success_threshold != null then 'success_threshold']: $.customized_success_threshold,

                mujoco_substeps: $.mujoco_substeps,
                
                goal_args: {
                    goal_state_paths: $.goal_state_paths,
                }
            },
            parameters: {
                simulation_params: {
                    task_object_configs: $.task_object_configs,
                    scene_object_configs: $.scene_object_configs,
                    shared_settings: $.shared_settings,
                },
            }
        }
    },
}
