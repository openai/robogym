local rainbow = (import "rainbow.libsonnet");

(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'rainbow/initial_state_20200427_231712.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ['rainbow/goal_state_20200428_204440.npz'],

    task_object_configs:: [
        rainbow['red'],
        rainbow['orange'],
        rainbow['yellow'],
        rainbow['green'],
        rainbow['blue'],
        rainbow['violet']
    ]
}
