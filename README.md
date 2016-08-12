## Robust Policy Search Algorithms

This repo has robust (to model parameters) variants for policy search algorithms. Current implimentations primarily look at episodic tasks and with emphasis on batch policy optimization using various forms of policy gradients.

This project builds on top of OpenAI gym and rllab. You need to set up those first before proceeding. The code structure is as follows:

### Things that have been implimented for you:
- **Environments:** Any environment file can be used which follows the structure of OpenAI gym. For MuJoCo tasks, you need to install mujoco-py which is shipped along with rllab. We have developed some environments which draw model parameters in the first step of stochastic MDP. See examples and read paper for more information on how to use them.
- **Policy:** For continuous control tasks, GaussianMLPPolicy can be chosen which is a neural network policy with gaussian exploration noise. For discrete action space, CategoricalMLPPolicy can be used. Both of these implimentations come from rllab.
- **Baselines:** LinearFeatureBaseline from rllab is tested and works correctly with this project.
- **Samplers:** Various samplers (both serial and parallel) for both a fixed environment as well as distributions of environments have been implimented by us.
- **Algorithms:** Currently, we have TRPO and REINFORCE. These have been implimented by us by modifying the basic structure provided by rllab.
- **Evaluations:** Functions for efficiently evaluating the performance of a policy on a given MDP or distribution of MDPs.

### Things you need to impliment:
- First copy the gym_env.py file from `base code` folder to `/path/to/rllab/rllab/envs/` and replace existing file
- Include the robustRL package into your pythonpath. `PYTHONPATH="/path/to/robustRL:$PYTHONPATH"` or change the `~/.bashrc` file.
- Ideally, you shouldn't have to touch any file other than `job_data.txt` and `MDP_funcs.py`
- Inside `MDP_funcs.py`, you need to write a function to generate the environment of your choice. Make sure this is compatible with OpenAI gym and that you have registered this environment with the gym modules. Also remember to add a function call within the `generate_environment` function in `MDP_funcs.py`
- **Note:** I recommend that you don't use a GPU for this, unless training convolutional layers. Modify the `.theanorc` file in your home directory to remove the GPU device set by default. Also uncomment the `theano.sandbox.cuda.unuse()` command in `algos.py` if you get a CUDA error.

Have a look at the example codes to get an idea of how the different functionality can be integrated for training. If you want to use just the training function without the wrappers, you can do this easily with just a for loop.

### Parallel sampling
Theano doesn't behave well with multiprocessing modules. If you run `python job_script.py`, you should see a bunch of worker processes starting up and then finishing. Running it in the background using nohup sometimes affects the writing to nohup.out and you may not see all processes starting up together, but start and end will happen alternatively. If this happens, see if multiple processes are spawned using htop command in terminal.

### To be included:
- DDPG algorithm of Lillicrap et al.
- Source domain adaptation using a Bayesian approach
- Bayesian state estimators (eg particle filters) for POMDP tasks

