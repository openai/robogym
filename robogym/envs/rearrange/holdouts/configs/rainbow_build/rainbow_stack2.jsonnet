local rainbow = (import "../rainbow.libsonnet");

(import "../base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'rainbow_build/initial_state_20200729_185445.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: [
        # Two stacked rainbow blocks with blue on top (largest -> smallest)
        'rainbow_build/goal_state_20200729_185515.npz',
    ],

    task_object_configs:: [
        rainbow['blue'],
        rainbow['violet']
    ],

    make_env +: {
        args +: {
            constants +: {
                success_threshold: {'obj_pos': 0.01, 'obj_rot': 0.2},
            },
        },
    },
}
