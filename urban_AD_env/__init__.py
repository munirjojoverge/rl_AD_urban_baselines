from gym.envs.registration import register

register(
    id='urban_AD-v1',
    entry_point='urban_AD_env.envs:MultiLaneEnv',
)

register(
    id='urban_AD-merge-v1',
    entry_point='urban_AD_env.envs:MergeEnv',
)

register(
    id='urban_AD-roundabout-v1',
    entry_point='urban_AD_env.envs:RoundaboutEnv',
)

register(
    id='urban_AD-continuous-v1',
    entry_point='urban_AD_env.envs:ContinuousEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20}
)

register(
    id='continuous-multi-env-v1',
    entry_point='urban_AD_env.envs:ContinuousMultiEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20}
)

register(
    id='parking-v1',
    entry_point='urban_AD_env.envs:ParkingEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20}
)