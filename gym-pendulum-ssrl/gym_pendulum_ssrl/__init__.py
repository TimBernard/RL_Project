from gym.envs.registration import register

register(
    id='PendulumSSRL-v0',
    entry_point='gym_pendulum_ssrl.envs:PendulumSSRLEnv',
)
register(
    id='PendulumSSRL-extrahard-v0',
    entry_point='gym_pendulum_ssrl.envs:PendulumSSRLExtraHardEnv',
)

