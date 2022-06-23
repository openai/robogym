local base = import 'raptor/experiments/base.jsonnet';
base.rl_training(
    env={"function": "raptor_robogym:wrap_env",
         "args": {"env": "robogym.envs.dactyl.reach:make_env"}},
    model="rl.learn.policy:make_mlp_policy",
    algorithm="rl.learn.actor_critic:PPO",
    use_distributed_communication=false,
    buffer_size=1024,
    batch_size=256,
    workers=8
)