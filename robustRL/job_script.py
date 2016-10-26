"""
    A simple wrapper to call the training modules.
    Useful when you want to run multiple jobs together in the
    background of a compute node

    Aravind Rajeswaran, 08/04/16
"""

import logging
logging.disable(logging.CRITICAL)
import sys
sys.dont_write_bytecode = True


from robustRL.algos import *
from robustRL.samplers import *
from robustRL.train_agent import *
from robustRL.utils import *
from MDP_funcs import *
np.random.seed(10)
rllab_set_seed(10)

job_set = []
data_file = open('job_data.txt', 'r')
for line in data_file:
    job_set.append(eval(line))

if __name__ == '__main__':
    t1 = timer.time()

    for job in job_set:
        print(" ============================================================")
        print("Started New Job : ", job['job_id'])
        print("Job specifications : \n", job)
        train_agent(**job)

    t2 = timer.time()
    print("Total time taken = ", t2-t1)
