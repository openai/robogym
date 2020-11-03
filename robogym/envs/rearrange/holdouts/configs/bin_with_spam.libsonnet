# Base config for bin_packing and bin_unpacking; consists of a bin with 3 spam cans.
(import "base.libsonnet") + {

    red::'1. 0.0 0.0 1.0',
    green::'0.0 1.0 0.0 1.0',
    blue::'0.0 0.0 1.0 1.0',
    orange::'1.0 0.5 0.0 1.0',

    local mat_wood = (import "../../materials/painted_wood.jsonnet"),

    task_object_configs:: [
        {
           count: 1,
           xml_path: 'holdouts/bin_packing/spam_can.xml',
           tag_args: {
               geom: {
                   rgba: $.orange,
               },
           },
           material_args: mat_wood,
        },
        {
           count: 1,
           xml_path: 'holdouts/bin_packing/spam_can.xml',
           tag_args: {
               geom: {
                   rgba: $.blue,
               },
           },
           material_args: mat_wood,
        },
        {
           count: 1,
           xml_path: 'holdouts/bin_packing/spam_can.xml',
           tag_args: {
               geom: {
                   rgba: $.red,
               },
           },
           material_args: mat_wood,
        },
        {
           count: 1,
           xml_path: 'holdouts/bin_packing/bin.xml',
           tag_args: {
               geom: {
                   rgba: $.green,
               },
               mesh: {
                   # Scaled so that the spam cans fit with a bit of room to spare.
                   scale: '0.00035 0.00035 0.00035',
               },
           },
           material_args: mat_wood,
        },
    ],
}
