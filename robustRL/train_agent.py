"""
    This file contains the function to train specified agent.
    Currently, you need to hardcode the algorithm inside this file.
    TO DO: Pull the algorithm declaration outside and into job_data.txt

    Aravind Rajeswaran, 08/04/16
"""

import logging
logging.disable(logging.CRITICAL)
import sys
sys.dont_write_bytecode = True
import os

from robustRL.algos import *
from robustRL.samplers import *
from robustRL.utils import save_plots_and_data
np.random.seed(10)
rllab_set_seed(10)

def train_agent(job_id,
    hidden_sizes = (32,32),
    max_kl = 0.01,
    niter = 250,
    gamma = 0.995,
    num_cpu = 'max',
    env_mode = 'train',
    mujoco_env = True,
    normalized_env = False,
    min_traj = 50,
    max_traj = 400,
    sub_sample = None,
    rwd_switch = 1500,
    policy = None,
    baseline = None,
    pol_restart_file = None,
    bas_restart_file = None,
    save_interim = True,
    save_paths = False,
    save_freq = 25,
    evaluate_test = False,
    plot_error_bar = False):

    # setup directories
    dirpath = 'experiment_%i' %job_id
    if os.path.isdir(dirpath) == False:
        os.mkdir(dirpath) 
    os.chdir(dirpath)
    result_file = ( open('results.txt', 'w') if pol_restart_file == None \
        else open('results.txt', 'a') )

    # Make appropriate environment (only for specs, not data)
    e = get_environment(env_mode)

    # Initialize policy and baseline correctly
    # 1) If policy or baseline is provided directly, they will be used
    # 2) If restart file provided, initialized from there
    # 3) If none of above, initialized randomly
    if policy == None:
        if pol_restart_file != None:
            policy = pickle.load(open(pol_restart_file, 'rb'))
        else:
            policy = GaussianMLPPolicy(env_spec=e.spec, hidden_sizes=hidden_sizes)

    if baseline == None:
        if bas_restart_file != None:
            baseline = pickle.load(open(bas_restart_file, 'rb'))
        else:
            baseline = LinearFeatureBaseline(env_spec=e.spec)
            if pol_restart_file != None:
                baseline_paths = sample_paths(20, policy, baseline, env_mode, 
                    mujoco_env=mujoco_env, normalized_env=normalized_env)
                baseline.fit(baseline_paths)

    # Create the agent
    agent = TRPO(e, policy, baseline, max_kl)

    def traj_schedule(iter, curr_return):
        if iter == 0 and pol_restart_file != None:
            return min_traj
        _slp = float(max_traj-min_traj); _slp = _slp/rwd_switch
        N = (min_traj if curr_return > rwd_switch \
            else min_traj + (rwd_switch-curr_return)*_slp )
        return min( int(np.ceil(N)), max_traj )

    # =======================================================================
    best_policy = copy.deepcopy(policy)
    best_perf = -1e8
    cum_num_ep = 0

    train_curve = best_perf*np.ones((niter,6))
    test_curve = np.zeros((niter,7))

    percentile_stats = best_perf*np.ones((niter, 7))

    for iter in range(niter):
        if train_curve[iter-1, 0] > best_perf:
            best_policy = copy.deepcopy(policy)
            best_perf = train_curve[iter-1, 0]

        num_traj = traj_schedule(iter, train_curve[iter-1, 0])
        cum_num_ep += num_traj

        train_curve[iter], percentile_stats[iter] = agent.train_step(num_traj, e.horizon, gamma, 
            env_mode=env_mode, num_cpu=num_cpu, save_paths=save_paths, idx=iter, 
            mujoco_env=mujoco_env, normalized_env=normalized_env, sub_sample=sub_sample)

        if evaluate_test:
            test_curve[iter] = policy_evaluation(policy, 'test', num_episodes=10)

        # save interim results
        if save_interim == True and iter % save_freq == 0 and iter > 0:
            save_plots_and_data(train_curve, test_curve, percentile_stats, policy, best_policy, 
                baseline, iter+1, mode='intermediate', evaluate_test=evaluate_test, 
                plot_error_bar=plot_error_bar)

        # Print results to console
        if iter == 0:
            print("Iter | Train MDP | Test MDP | Best (on Train) | num_traj | Episodes so far \n")
            result_file.write(" Iter | Train MDP | Test MDP | Best (on Train) | Episodes so far \n")
        print("[", timer.asctime( timer.localtime(timer.time()) ), "]", iter, \
            train_curve[iter,0], test_curve[iter,0], best_perf, num_traj, cum_num_ep)
        result_file.write("%3i %4.2f %4.2f %4.2f %3i %6i \n" % (iter, train_curve[iter,0],
            test_curve[iter,0], best_perf, num_traj, cum_num_ep) )

    save_plots_and_data(train_curve, test_curve, percentile_stats, policy, best_policy, 
        baseline, niter, mode='final', evaluate_test=evaluate_test, plot_error_bar=plot_error_bar)
    result_file.close()
