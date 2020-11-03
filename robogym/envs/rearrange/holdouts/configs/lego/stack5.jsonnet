(import "lego.libsonnet") + {

    use_easy_piece:: false,
    initial_state_path:: 'lego_stack/initial_state_20200728_183601.npz',
    goal_state_paths:: ['lego_stack/goal_state_20200728_172514.npz'],
              
    all_pieces:: [
       {color: '1.0 0.0 0.0 1.0', cnt: 1},
       {color: '0.0 1.0 0.0 1.0', cnt: 1},
       {color: '0.0 0.0 1.0 1.0', cnt: 1},
       {color: '0.0 1.0 1.0 1.0', cnt: 1},
       {color: '1.0 0.0 1.0 1.0', cnt: 1},
    ],
}

