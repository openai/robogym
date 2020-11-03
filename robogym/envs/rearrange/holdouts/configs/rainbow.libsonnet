local mat_wood = (import "../../materials/painted_wood.jsonnet");

{
	'red': {
       xml_path: 'holdouts/rainbow/rainbow_part5.xml',
       tag_args: {
           geom: {
               rgba: '1. 0.0 0.0 1.0',
           },
           mesh: {
               scale: '0.00102 0.00102 0.00102',
           },
       },
       material_args: mat_wood,
    },
    'orange': {
       xml_path: 'holdouts/rainbow/rainbow_part4.xml',
       tag_args: {
           geom: {
               rgba: '0.95 0.4745 0.086 1.0',
           },
           mesh: {
               scale: '0.001 0.001 0.001',
           },
       },
       material_args: mat_wood,
    },
    'yellow': {
       xml_path: 'holdouts/rainbow/rainbow_part3.xml',
       tag_args: {
           geom: {
               rgba: '1.0 1.0 0. 1.0',
           },
           mesh: {
               scale: '0.001 0.001 0.001',
           },
       },
       material_args: mat_wood,
    },
    'green': {
       xml_path: 'holdouts/rainbow/rainbow_part2.xml',
       tag_args: {
           geom: {
               rgba: '0.0 1.0 0.0 1.0',
           },
           mesh: {
               scale: '0.001 0.001 0.001',
           },
       },
       material_args: mat_wood,
    },
    'blue': {
       xml_path: 'holdouts/rainbow/rainbow_part1.xml',
       tag_args: {
           geom: {
               rgba: '0.215 0.513 1.0 1.0',
           },
           mesh: {
               scale: '0.001 0.001 0.001',
           },
       },
       material_args: mat_wood,
    },
    'violet': {
       xml_path: 'holdouts/rainbow/rainbow_part0.xml',
       tag_args: {
           geom: {
               rgba: '0.28 0.08 0.67 1.0',
           },
           mesh: {
               scale: '0.001 0.001 0.001',
           },
       },
       material_args: mat_wood,
    },
}