(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'tangram/initial_state_20200508_182131.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ['tangram/goal_state_20200508_183958.npz'],

    light_black::'0.2 0.2 0.2 1.0',
    tan::'0.82 0.76 0.55 1.0',
    blue::'0.23 0.64 0.83 1.0',

    object_scale:: '0.001 0.001 0.001',

    local mat = (import "../../materials/tangram.jsonnet"),

    task_object_configs:: [
        {
           xml_path: 'holdouts/tangram/tangram_large_triangle.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/tangram/tangram_medium_triangle.xml',
           count: 1,
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/tangram/tangram_small_triangle.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/tangram/tangram_square.xml',
           count: 1,
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/tangram/tangram_parallelogram.xml',
           count: 1,
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
    ]
}
