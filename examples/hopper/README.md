### Example 1
In this example, we train a (64,64) net policy on the standard hopper environment and test it on a heavy hopper (twice the torso mass). This is just to show an example. Typically, we need to train longer (for approx 250 or 300 iterations) and with a lower learning rate (max_kl=0.01).

Here, we see that training on the standard hopper which has a torso mass of approx 3.5 (calculated using density=1000 in MuJoCo) and testing it on a hopper with heavier torso (mass=7, twice of standard) fails badly. However, if we train on an ensemble, the performance on test MDPs improve dramatically. See paper for more details.

To run:
```
source activate rllab
python job_script.py
```
or for background process,
```
source activate rllab
nohup python job_script 2> log &
```

Check the main README file to make sure you copied all the files to appropriate locations.

