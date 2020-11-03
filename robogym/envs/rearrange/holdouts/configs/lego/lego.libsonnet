(import "../base.libsonnet") + {

    use_easy_piece:: false,

    mujoco_timestep:: 0.001,
    
    shared_settings:: if $.use_easy_piece then 'holdouts/lego/easy_defaults.xml' else 'holdouts/lego/defaults.xml',

    customized_success_threshold:: {'obj_pos': 0.04, 'obj_rot': 0.1, 'rel_dist': 0.002},
    
    task_object_configs:: [
        {
           xml_path: 'holdouts/lego/duplo2x4.xml',
           count: key['cnt'],
           tag_args: {
               geom: {
                   rgba: key['color'],
               },
           },
         }
         for key in $.all_pieces
    ]
}
