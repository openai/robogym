(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'ball_capture/initial_state_20200504_010248.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ['ball_capture/goal_state_20200504_013035.npz'],

    red::'1. 0.0 0.0 1.0',
    green::'0.0 1.0 0.0 1.0',
    blue::'0.0 0.0 1.0 1.0',

    local mat_wood = (import "../../materials/painted_wood.jsonnet"),
    local mat_rubber_ball = (import "../../materials/rubber-ball.jsonnet"),

    task_object_configs:: [
        {
           count: 2,
           xml_path: 'holdouts/ball_capture/cylinder.xml',
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
           },
           material_args: mat_wood,
        },
        {
           count: 2,
           xml_path: 'holdouts/ball_capture/cylinder.xml',
           tag_args: {
               geom: {
                   rgba: $.green,
               },
           },
           material_args: mat_wood,
        },
        {
           count: 2,
           xml_path: 'holdouts/ball_capture/ball.xml',
           tag_args: {
               geom: {
                   rgba: $.red,
               },
           },
           material_args: mat_rubber_ball,
        },
    ],
}
