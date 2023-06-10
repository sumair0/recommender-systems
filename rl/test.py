import argparse
import os
from datetime import datetime as dt

import gym

import recommendation_env


# env = gym.make(recommendation_env.RecoEnv.id, **recommendation_env.import_data_for_env())

env = recommendation_env.RecoEnv(**recommendation_env.import_data_for_env())

print ("Here")