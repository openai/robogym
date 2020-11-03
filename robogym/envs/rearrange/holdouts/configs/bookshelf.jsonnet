(import "base.libsonnet") + {

    # Comment this out when using bin/create_holdout
    initial_state_path:: 'bookshelf/initial_state_20200731_163944.npz',

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ['bookshelf/goal_state_20200803_161116.npz'],

    local mat_wood = (import "../../materials/painted_wood.jsonnet"),

    task_object_configs:: [
        {
            xml_path: 'holdouts/bookshelf/plane.xml',
            material_args: mat_wood,
            count: 2
        },
        {
            xml_path: 'holdouts/bookshelf/bookshelf.xml',
            material_args: mat_wood,
        },
    ]
}