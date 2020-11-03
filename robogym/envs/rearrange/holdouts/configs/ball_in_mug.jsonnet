(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: "ball_in_mug/initial_state_20200804_153252.npz",

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ["ball_in_mug/goal_state_20200804_153422.npz"],

    red::'1. 0.0 0.0 1.0',
    blue::'0.0 0.0 1.0 1.0',

    local mat_wood = (import "../../materials/painted_wood.jsonnet"),
    local mat_rubber_ball = (import "../../materials/rubber-ball.jsonnet"),

    task_object_configs:: [
        {
           count: 1,
           xml_path: 'holdouts/ball_capture/ball.xml',
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
           },
           material_args: mat_rubber_ball,
        },
        {
           count: 1,
           xml_path: 'holdouts/ball_in_mug/mug.xml',
           tag_args: {
               geom: {
                   rgba: $.red,
               },
           },
           material_args: mat_wood,
        },
    ],
}
