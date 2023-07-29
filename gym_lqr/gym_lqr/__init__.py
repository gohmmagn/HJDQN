from gymnasium.envs.registration import register

register(
    id='LinearQuadraticRegulator20D-v0',
    entry_point='gym_lqr.envs:LinearQuadraticRegulator20DEnv',
    max_episode_steps=200,
)

register(
    id='Linear1dPDEEnv-v0',
    entry_point='gym_lqr.envs:Linear1dPDEEnv',
    max_episode_steps=200,
)

register(
    id='Linear2dPDEEnv-v0',
    entry_point='gym_lqr.envs:Linear2dPDEEnv',
    max_episode_steps=200,
)

register(
    id='NonLinearPDEEnv-v0',
    entry_point='gym_lqr.envs:NonLinearPDEEnv',
    max_episode_steps=200,
)