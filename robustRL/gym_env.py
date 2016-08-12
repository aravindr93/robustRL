from __future__ import print_function
from __future__ import absolute_import

import gym
import gym.envs
import gym.spaces
from gym.monitoring import monitor
import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import logger
import logging
import numpy as np

def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    def __call__(self, count):
        return monitor.capped_cubic_video_schedule(count)


class FixedIntervalVideoSchedule(object):

    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None):

## following lines modified by me (correspondingly commented out below) to suppress the warning messages
        if log_dir is None and logger.get_snapshot_dir() is not None:
            log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")

# *********************
#        if log_dir is None:
#            if logger.get_snapshot_dir() is None:
#                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
#            else:
#                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
# *********************

        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        self.env = env
        self.env_id = env.spec.id

        monitor.logger.setLevel(logging.CRITICAL)

        if log_dir is None:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env.monitor.start(log_dir, video_schedule)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit
        self._log_dir = log_dir

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    # I have writtin the method evaluate_policy for easy use
    def evaluate_policy(self, policy, num_episodes=5, horizon=1e6, gamma=1, visual=False,
        percentile=[], get_full_dist=False):
        horizon = min(horizon, self._horizon)
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        for ep in range(num_episodes):
            o = self.reset()
            t, done = 0, False
            while t < horizon and done != True:
                if visual == True:
                    self.render()
                a = policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1
                #if visual == True and done == True:
                #    s = self.env.state_vector()
                #    posafter,height,ang = self.env.model.data.qpos[0:3,0]
                #    print("Termination reason : \n", np.isfinite(s).all(), (np.abs(s[2:]) < 100).all(), \
                #        (height > .7), (abs(ang) < .2) )
        
        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        full_dist = []

        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        if get_full_dist == True:
            full_dist = ep_returns

        return [base_stats, percentile_stats, full_dist]


    def terminate(self):
        if self.monitoring:
            self.env.monitor.close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)

