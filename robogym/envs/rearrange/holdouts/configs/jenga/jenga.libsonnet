(import "../base.libsonnet") + {

    tan::'0.82 0.71 0.55 1.0',

    mujoco_timestep:: 0.001,

    num_pieces:: 6,

    local mat_wood = (import "../../../materials/painted_wood.jsonnet"),

    customized_success_threshold:: {'obj_pos': 0.03, 'obj_rot': 0.1},

    make_env +: {
        args +: {
            constants +: {
                goal_args +: {
                     "rot_dist_type": "mod180",
                },
            },
        },
    },

    task_object_configs:: [
        {
           count: $.num_pieces,
           xml_path: 'holdouts/jenga/piece.xml',
           tag_args: {
               geom: {
                   rgba: $.tan,
               },
           },
           material_args: mat_wood,
        },
    ],
}

