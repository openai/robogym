(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: './dominoes/initial_state_20200619_134646.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ['./dominoes/domino_goal_state_20200716.npz'],

    tan::'0.82 0.71 0.55 1.0',

    local mat_wood = (import "../../materials/painted_wood.jsonnet"),

    task_object_configs:: [
        {
           count: 20,
           xml_path: 'holdouts/dominoes/domino.xml',
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
           },
           material_args: mat_wood,
        },
    ],
}
