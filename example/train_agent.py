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

from algos import *
from samplers import *
from utils import save_plots_and_data
np.random.seed(10)
rllab_set_seed(10)

def train_agent(job_id,
    hidden_sizes = (32,32),
    max_kl = 0.01,
    niter = 250,
    gamma = 0.995,
    num_cpu = 'max',
    restart_file = None,
    save_interim = True,
    save_freq = 25,
    evaluate_test = False,
    plot_error_bar = False):

    # setup directories
    dirpath = 'experiment_%i' %job_id
    if os.path.isdir(dirpath) == False:
        os.mkdir(dirpath) 
    os.chdir(dirpath)
    result_file = ( open('results.txt', 'w') if restart_file == None \
        else open('results.txt', 'a') )

    e = get_environment('train')
    policy = GaussianMLPPolicy(env_spec=e.spec, hidden_sizes=hidden_sizes)
    baseline = LinearFeatureBaseline(env_spec=e.spec)
    if restart_file != None:
        policy = pickle.load(open(restart_file, 'rb'))
    agent = TRPO(e, policy, baseline, max_kl)

    def traj_schedule(iter, curr_return):
        N = (50 if curr_return > 1500 else 50+(1500-curr_return)*350.0/1500)
        return int(np.ceil(N))

    # =======================================================================
    train_curve = np.zeros((niter,5))
    test_curve = np.zeros((niter,5))

    best_policy = copy.deepcopy(policy)
    best_perf = 0
    cum_num_ep = 0

    for iter in range(niter):
        if train_curve[iter-1, 0] > best_perf:
            best_policy = copy.deepcopy(policy)
            best_perf = train_curve[iter-1, 0]

        num_traj = traj_schedule(iter, train_curve[iter-1, 0])
        cum_num_ep += num_traj

        train_curve[iter] = agent.train_step(num_traj, e.horizon,
            gamma, env_mode='train', num_cpu=num_cpu)

        if evaluate_test:
            test_curve[iter] = policy_evaluation(policy, 'test', num_episodes=10)

        # save interim results
        if save_interim == True and iter % save_freq == 0 and iter > 0:
            save_plots_and_data(train_curve, test_curve, policy, best_policy, iter+1,
                mode='intermediate', evaluate_test=evaluate_test, 
                plot_error_bar=plot_error_bar)

        # Print results to console
        if iter == 0:
            print "Iter | Train MDP | Test MDP | Best (on Train) | num_traj | Episodes so far \n"
            result_file.write(" Iter | Train MDP | Test MDP | Best (on Train) | Episodes so far \n")
        print "[", timer.asctime( timer.localtime(timer.time()) ), "]", iter, \
            train_curve[iter,0], test_curve[iter,0], best_perf, num_traj, cum_num_ep
        result_file.write("%3i %4.2f %4.2f %4.2f %3i %6i \n" % (iter, train_curve[iter,0],
            test_curve[iter,0], best_perf, num_traj, cum_num_ep) )

    save_plots_and_data(train_curve, test_curve, policy, best_policy, niter,
        mode='final', evaluate_test=evaluate_test, plot_error_bar=plot_error_bar)
    result_file.close()
