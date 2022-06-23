local base = import 'raptor/experiments/base.jsonnet';
base.rl_training(
    env={"function": "raptor_robogym:wrap_env",
         "args": {"env": {"function": "robogym.envs.dactyl.reach:make_env", "args": {"constants": {"randomize": false}}},}},
    model={"function": "rl.learn.policy:make_mlp_policy", "args": {"normalize": false}},
    algorithm={"function": "rl.learn.actor_critic:PPO", "args": {"advantage_normalize": true}},
    optimizer_fn=base.optimizer_fn(optim_stepsize=3e-5),
    use_distributed_communication=false,
    optimizer_clip_norm=1.0,
    buffer_size=256,
    max_staleness=0,
    batch_size=256,
    workers=8
)