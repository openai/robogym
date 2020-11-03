(import "jenga.libsonnet") + {
    num_pieces:: 6,
    initial_state_path:: './jenga/initial_state_20200803_103914.npz',
    goal_state_paths:: ['./jenga/goal_state_20200803_103840.npz'],
    customized_success_threshold:: {'obj_pos': 0.02, 'obj_rot': 0.1},
}
