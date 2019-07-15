from gym.envs.registration import register

register(
    id='ParkQOptEnv-v0',
    entry_point='drl_fw.envs.park_qopt_env:ParkQOptEnv'
)

register(
    id='RayParkQOptEnv-v0',
    entry_point='drl_fw.envs.ray_park_qopt_env:RayParkQOptEnv'
)
