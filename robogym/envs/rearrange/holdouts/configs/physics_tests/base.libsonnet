(import "../base.libsonnet") + {
    make_env +: {
        args +: {
            constants +: {
                stabilize_objects: false,
            },
            parameters +: {
                n_random_initial_steps: 0,
            }
        }
    },

    # We use `mat::` so that we can easily override this with other materials.
    mat:: (import "../../../materials/painted_wood.jsonnet"),

    xml_path:: 'primitives/box.xml',

    rgba:: "1 0 0 1",

    quat:: "1 0 0 0",

    count:: 1,

    task_object_configs:: [
        {
            count: $.count,
            xml_path: $.xml_path,
            material_args: $.mat,
            tag_args: {
                geom: {
                    rgba: $.rgba,
                    quat: $.quat,
               },
               mesh: {
                   scale: '0.001 0.001 0.001',
               },
            },
        },
    ],
}
