(import "lego.libsonnet") + {

    use_easy_piece:: false,
    initial_state_path:: 'stack2/initial_state_20200729_111147.npz',
    goal_state_paths:: ['stack2/goal_state_20200729_111050.npz'],
              
    all_pieces:: [
       {color: '1.0 0.0 0.0 1.0', cnt: 1},
       {color: '0.0 1.0 0.0 1.0', cnt: 1},
    ],
}

