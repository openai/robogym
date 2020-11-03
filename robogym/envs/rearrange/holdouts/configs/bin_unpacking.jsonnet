(import "bin_with_spam.libsonnet") + {
    # Comment this out when using bin/create_holdout
    initial_state_path:: "bin_packing/goal_state_20200804_152834.npz",

    # Comment this out when using bin/create_holdout
    goal_state_paths:: ["bin_packing/initial_state_20200804_152516.npz"],
}
