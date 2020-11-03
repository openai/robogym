local rainbow = (import "../rainbow.libsonnet");

(import "../base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'rainbow_build/initial_state_20200729_184825.npz',

    # Comment this out when using bin/create_holdout
    # TODO: Physics is still causing the purple block to jitter, and so
    # the overall structure isn't super stable. As we improve physics try
    # tuning the friction coefficient and also try balancing the orange and
    # yellow blocks to make this holdout more complex.
    goal_state_paths:: [
        # Red and purple blocks balancing on upside down green/blue arcs
        'rainbow_build/goal_state_20200729_184920.npz'
    ],

    task_object_configs:: [
        rainbow['red'],
        rainbow['green'],
        rainbow['blue'],
        rainbow['violet']
    ],

    make_env +: {
        args +: {
            constants +: {
                success_threshold: {'obj_pos': 0.03, 'obj_rot': 0.2},
            },
        },
    },
}
