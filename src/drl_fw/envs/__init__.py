from gym.envs.registration import register
from ray.tune.registry import register_env
from drl_fw.envs.ray_park_qopt_env import RayParkQOptEnv

register(
    id='ParkQOptEnv-v0',
    entry_point='drl_fw.envs.park_qopt_env:ParkQOptEnv'
)

register_env("RayParkQOptEnv-v0", lambda config: RayParkQOptEnv())


# register(
#     id='RayParkQOptEnv-v0',
#     entry_point='drl_fw.envs.ray_park_qopt_env:RayParkQOptEnv'
# )
