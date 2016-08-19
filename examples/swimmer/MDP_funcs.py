"""
    Add the environment you wish to train here
"""

import gym
import numpy as np
from rllab.envs.gym_env import GymEnv

# ===================================================================
# Functions used in training and testing phase

def make_train_MDP():
    return _standard_swimmer()

def make_test_MDP():
    return _heavy_hopper()

# ===================================================================
# Local functions to create envs

def _standard_swimmer():
    return GymEnv('Swimmer-v1')

# =======================================================================================
# Generate environment corresponding to the given mode

def get_environment(env_mode):

    modes = ['train', 'test']    

    if env_mode == 'train':
        env = make_train_MDP()
    #elif env_mode == 'test':
    #    env = make_test_MDP()
    else:
        print "ERROR: Unknown environment mode specified. Allowed modes are ", modes

    return env
