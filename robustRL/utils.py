"""
    Other utility functions go here.
    Currently has only functions to make various plots.

    Aravind Rajeswraan, 08/04/16
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def save_plots_and_data(train_curve, 
    test_curve,
    percentile_stats,
    policy,
    best_policy,
    baseline,
    iter,
    mode = 'intermediate',
    evaluate_test = False,
    plot_error_bar = True):

    if mode == 'final':
        with open('train_curve.txt', 'ab') as f:
            np.savetxt(f, train_curve, fmt='%4.3f')

        with open('train_percentile_stats.txt', 'wb') as f:
            np.savetxt(f, percentile_stats, fmt='%4.3f')

        if evaluate_test:
            with open('test_curve.txt', 'ab') as f:
                np.savetxt(f, test_curve, fmt='%4.3f')
            test_curve = np.loadtxt('test_curve.txt')

        train_curve = np.loadtxt('train_curve.txt')
        plt1_file_name = 'train_curve.png'
        plt2_file_name = 'train_test_comparison.png'

        policy_file = 'final_policy.pickle'
        baseline_file = 'final_baseline.pickle'

    elif mode == 'intermediate':
        with open('train_curve_interim.txt', 'wb') as f:
            np.savetxt(f, train_curve[0:iter], fmt='%4.3f')

        with open('train_percentile_stats.txt', 'wb') as f:
            np.savetxt(f, percentile_stats[0:iter], fmt='%4.3f')
        
        if evaluate_test:
            with open('test_curve_interim.txt', 'wb') as f:
                np.savetxt(f, test_curve[0:iter], fmt='%4.3f')
        
        train_curve = train_curve[0:iter]
        test_curve = test_curve[0:iter]
        plt1_file_name = 'train_curve_interim.png'
        plt2_file_name = 'train_test_comparison_interim.png'

        policy_file = 'policy_%i.pickle' %iter
        baseline_file = 'baseline_%i.pickle' %iter

    else:
        print("ERROR: Choose only intermediate or final for mode")


    #print train_curve, test_curve

    # Main learning curve
    #y_max = max(0, np.amax(train_curve[:,3]))
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(train_curve[:,0], label='Learning Curve')
    if plot_error_bar:
        #ax.plot(train_curve[:,0] + 1.96*train_curve[:,1]/np.sqrt(train_curve[:,4]), 
            #label='$\mu + 1.96\sigma$', ls='--')
        #ax.plot(train_curve[:,0] - 1.96*train_curve[:,1]/np.sqrt(train_curve[:,4]), 
            #label='$\mu - 1.96\sigma$', ls='--')
        ax.plot(train_curve[:,0] + train_curve[:,1], label='$\mu + \sigma$', ls='--')
        ax.plot(train_curve[:,0] - train_curve[:,1], label='$\mu - \sigma$', ls='--')
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('Learning Curve')
    #ax.set_ylim([0, y_max])
    ax.set_xlim([0, iter-1])
    ax.legend(loc=2)
    plt.savefig(plt1_file_name)
    plt.close()

    # Comparison plot
    if evaluate_test:
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(train_curve[:,0], label='Train MDP')
        ax.plot(test_curve[:,0], label='Test MDP')
        plt.xlabel('Iterations')
        plt.ylabel('Returns')
        plt.title('Comparison')
        ax.set_xlim([0, iter-1])
        ax.legend(loc=2)
        plt.savefig(plt2_file_name)  
        plt.close()     

    # Save the policies and baseline
    if os.path.isdir('iterations') == False:
        os.mkdir('iterations')
    os.chdir('./iterations/')
    pickle.dump(policy, open(policy_file, 'wb'))
    pickle.dump(best_policy, open('best_policy.pickle', 'wb'))
    pickle.dump(baseline, open(baseline_file, 'wb'))
    os.chdir('..')


def save_paths(paths, idx=0):
    # Saves the given paths with name: ./iterations/paths_<idx>.pickle
    if os.path.isdir('iterations') == False:
        os.mkdir('iterations')
    os.chdir('./iterations/')
    paths_file = 'paths_%i.pickle' %idx
    pickle.dump(paths, open(paths_file, 'wb'))
    os.chdir('..')
