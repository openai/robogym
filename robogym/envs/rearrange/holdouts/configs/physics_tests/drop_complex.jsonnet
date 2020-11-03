(import "base.libsonnet") + {
    initial_state_path:: 'physics_tests/drop_complex/initial_state_20200625_165611.npz',

    task_object_configs +:: [
        {
            count: 20,
            xml_path: 'holdouts/chess/chess_knight.xml',
            material_args: $.mat,
            tag_args: {
                geom: {
                    rgba: $.rgba,
                },
                mesh: {
                   scale: '0.001 0.001 0.001',
               },
            },
        },
    ],
}
