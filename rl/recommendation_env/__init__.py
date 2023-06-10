from gym.envs.registration import register
from recommendation_env.envs.reco_env import RecoEnv
from recommendation_env.utils import import_data_for_env , evaluate


register(
    id=RecoEnv.id,
    entry_point='recommendation_env.envs:RecoEnv',
    max_episode_steps=1000000,
    nondeterministic=False
)
