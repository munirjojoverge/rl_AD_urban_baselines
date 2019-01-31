from gym.envs.registration import register

register(
    id='urban-v0',
    entry_point='urban_AD_env.envs:MultiLaneEnv',
)

register(
    id='urban-merge-v0',
    entry_point='urban_AD_env.envs:MergeEnv',
)

register(
    id='urban-roundabout-v0',
    entry_point='urban_AD_env.envs:RoundaboutEnv',
)
