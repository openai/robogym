(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'sample/initial_state_20200422_231828.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ['sample/goal_state_20200422_232013.npz'],

    local mat_wood = (import "../../materials/painted_wood.jsonnet"),

    scene_object_configs:: [
        {
           xml_path: 'chess/chessboard.xml',
           pos: [0.5, 0.5, 0.001],
        },
    ],

    task_object_configs:: [
        {
           xml_path: 'object/tests/can.xml',
           tag_args: {
               mesh: {
                   # Override mesh scale.
                   scale: '0.5 0.5 0.5',
               },
           },

        },
        {
           xml_path: 'object/tests/block.xml',
           tag_args: {
               geom: {
                   # Override geom size.
                   size: '0.025 0.025 0.025',
               },
           },

           material_args: mat_wood,

           count: 2,
        },
    ]
}
