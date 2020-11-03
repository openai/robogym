local rainbow = (import "../rainbow.libsonnet");

(import "../base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'rainbow_build/initial_state_20200729_200615.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: [
        # Four stacked rainbow blocks with green on top (smallest -> largest)
        'rainbow_build/goal_state_20200729_200650.npz',
    ],

    task_object_configs:: [
        rainbow['red'],
        rainbow['orange'],
        rainbow['yellow'],
        rainbow['green']
    ],

    make_env +: {
        args +: {
            constants +: {
                success_threshold: {'obj_pos': 0.03, 'obj_rot': 0.2},
            },
        },
    },
}
