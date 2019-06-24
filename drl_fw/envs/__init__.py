from gym.envs.registration import register

register(
    id='ParkQueryOptimizer-v0',
    entry_point='drl_fw.envs.query_optimizer:ParkQueryOptimizer'
)
