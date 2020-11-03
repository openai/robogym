(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'chess/initial_state_20200504_163008.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ['chess/goal_state_20200504_163009.npz'],

    light_black::'0.2 0.2 0.2 1.0',
    tan::'0.82 0.76 0.55 1.0',

    object_scale:: '0.001 0.001 0.001',

    local mat = (import "../../materials/chess.jsonnet"),

    scene_object_configs:: [
        {
            xml_path: 'holdouts/chess/chessboard.xml',
            tag_args: {
                geom: {
                    pos: "0.7 0.5 0.001"
                },
            },
        },
    ],

    task_object_configs:: [
        {
           xml_path: 'holdouts/chess/chess_pawn.xml',
           count: 8,
           tag_args: {
               geom: {
                   rgba: $.light_black,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_pawn.xml',
           count: 8,
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_rook.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.light_black,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_rook.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_bishop.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.light_black,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_bishop.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_knight.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.light_black,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_knight.xml',
           count: 2,
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_king.xml',
           tag_args: {
               geom: {
                   rgba: $.light_black,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_king.xml',
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_queen.xml',
           tag_args: {
               geom: {
                   rgba: $.light_black,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
        {
           xml_path: 'holdouts/chess/chess_queen.xml',
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
               mesh: {
                   scale: $.object_scale,
               },
           },
           material_args: mat,
        },
    ]
}
