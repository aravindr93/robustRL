"""
    Sampler functions to be used with the policy search algorithms

    Aravind Rajeswaran, 08/04/16
"""

import numpy as np
import copy
import multiprocessing as mp

from rllab.misc import tensor_utils

from MDP_funcs import *
# above MDP_funs is local copy

# =======================================================================================
# Functions for sampling paths

def sample_paths(N, 
    policy, 
    baseline, 
    env_mode='train', 
    T=1e6, 
    gamma=1,
    mujoco_env=True, 
    normalized_env=False,
    env=None):
    # Directly specifying env works only when sampling in series

    # set random seed (needed for multiprocessing)
    np.random.seed()

    if env == None:
        env = get_environment(env_mode)
    T = min(T, env.horizon)
    T = max(1, T)  
    # sometimes, env is not initialized correctly in multiprocessing
    # this is just a sanity check and step size should essentially be zero.

    print("####### Worker started #######")

    paths = []

    for ep in range(N):
        
        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []
        qpos = []
        qvel = []

        o = env.reset()
        if mujoco_env == True:
            if normalized_env:
                qpos.append(env.wrapped_env.env.model.data.qpos.reshape(-1))
                qvel.append(env.wrapped_env.env.model.data.qvel.reshape(-1))
            else:
                qpos.append(env.env.model.data.qpos.reshape(-1))
                qvel.append(env.env.model.data.qvel.reshape(-1))
        done = False
        t = 0

        while t < T and done != True:
            a, agent_info = policy.get_action(o)
            next_o, r, done, env_info = env.step(a)
            observations.append(env.observation_space.flatten(o))
            actions.append(env.action_space.flatten(a))
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            if mujoco_env == True:
                if normalized_env:
                    qpos.append(env.wrapped_env.env.model.data.qpos.reshape(-1))
                    qvel.append(env.wrapped_env.env.model.data.qvel.reshape(-1))
                else:
                    qpos.append(env.env.model.data.qpos.reshape(-1))
                    qvel.append(env.env.model.data.qvel.reshape(-1))
            o = next_o
            t += 1

        # make a path dictionary
        # Also store the path belief and env data used in the trajectory
        #try:
        #    path_belief = env.env.belief
        #except Exception as e:
        #    path_belief = str(e)
        # path_model = env.env

        qpos_flat = tensor_utils.stack_tensor_list(qpos)
        qvel_flat = tensor_utils.stack_tensor_list(qvel)

        path = dict(
            observations=tensor_utils.stack_tensor_list(observations),
            actions=tensor_utils.stack_tensor_list(actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            qpos=qpos_flat,
            qvel=qvel_flat,
            #path_belief=path_belief,
            #path_model=path_model,
        )

        # TODO: Storing the path model is too space inefficient. Need to find alternative
        
        # compute returns using the path
        path_baseline = baseline.predict(path)
        advantages = []
        returns = []
        return_so_far = 0
        for t in range(len(rewards) - 1, -1, -1):
            return_so_far = rewards[t] + gamma * return_so_far
            returns.append(return_so_far)
            advantage = return_so_far - path_baseline[t]
            advantages.append(advantage)

        # advantages and returns are stored backward in time
        advantages = np.array(advantages[::-1])
        returns = np.array(returns[::-1])
        
        # normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        path["advantages"] = advantages
        path["returns"] = returns

        paths.append(path)

    #print "Env body_mass : ", env.env.model.body_mass[1]
    print("====== Worker finished ======")

    return paths


def _sample_paths_star(args_list):
    """ Constructor function to pass an args_list.
        Can call pool.map on this function """
    return sample_paths(*args_list)


def sample_paths_parallel(N,
    policy,
    baseline,
    env_mode='train',
    T=1e6, gamma=1,
    num_cpu=None,
    max_process_time=120,
    max_timeouts=4,
    mujoco_env=True,
    normalized_env=False):
    
    if num_cpu == None or num_cpu == 'max':
        num_cpu = mp.cpu_count()
    elif num_cpu == 1:
        return sample_paths(N, policy, baseline, evn_mode, T, gamma, mujoco_env, normalized_env)
    else:
        num_cpu = min(mp.cpu_count(), num_cpu)       

    paths_per_cpu = int(np.ceil(N/num_cpu))
    args_list = [paths_per_cpu, policy, baseline, env_mode, T, gamma, mujoco_env, normalized_env]

    results = _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts)

    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    return paths


def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts):
    
    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    parallel_runs = [pool.apply_async(_sample_paths_star, args=(args_list,)) for _ in range(num_cpu)]

    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()        
        return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts-1)

    pool.close()
    pool.terminate()
    pool.join()  
    return results

# =======================================================================================
# Functions for performance evaluation

def policy_evaluation(policy, 
    env_mode='train',
    num_episodes=10,
    horizon=1e6,
    visual=False,
    gamma=1):
    # TODO: Add functionality to sample parallel paths and evaluate policy

    env = get_environment(env_mode)
    horizon = min(env.horizon, horizon)

    ep_returns = np.zeros(num_episodes)

    for ep in range(num_episodes):
        o = env.reset()
        t = 0
        done = False
        while t < horizon and done != True:
            if visual == True:
                env.render()
            a = policy.get_action(o)[0]
            o, r, done, _ = env.step(a)
            ep_returns[ep] += (gamma ** t) * r
            t += 1

    mean_eval = np.mean(ep_returns)
    std_eval  = np.std(ep_returns)
    min_eval  = np.amin(ep_returns)
    max_eval  = np.amax(ep_returns)
    soft_min  = np.percentile(ep_returns,15)
    soft_max  = np.percentile(ep_returns,85)

    return (mean_eval, std_eval, min_eval, max_eval, soft_min, soft_max, num_episodes)
