local rainbow = (import "../rainbow.libsonnet");

(import "../base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'rainbow_build/initial_state_20200728_111736.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: [
        # Six stacked rainbow blocks with purple on top (smallest -> largest)
        'rainbow_build/goal_state_20200728_121612.npz',
    ],

    task_object_configs:: [
        rainbow['red'],
        rainbow['orange'],
        rainbow['yellow'],
        rainbow['green'],
        rainbow['blue'],
        rainbow['violet']
    ],

    make_env +: {
        args +: {
            constants +: {
                success_threshold: {'obj_pos': 0.04, 'obj_rot': 0.2},
            },
        },
    },
}
