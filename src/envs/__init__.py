from gym.envs.registration import register

register(
    id='ParkQueryOptimizer-v0',
    entry_point='envs.query_optimizer:ParkQueryOptimizer'
)
