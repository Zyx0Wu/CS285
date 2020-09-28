Experiment 1: 
```
python cs285/scripts/experiment1.py --env_name CartPole-v0 -n 100 -b 1000
```
```
python cs285/scripts/experiment1.py --env_name CartPole-v0 -n 100 -b 5000
```
Experiment 2: 
```
python cs285/scripts/experiment2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -rtg
```
Experiment 3: 
```
python cs285/scripts/experiment3.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005
```
Experiment 4: 
```
python cs285/scripts/experiment4.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32
```
Bonus: 
```
python cs285/scripts/bonus.py --env_name Walker2d-v2 --ep_len 1000 --discount 0.99 --lambda 0.95 -n 100 -b 10000 -lr 0.005 -rtg --nn_baseline --eval_batch_size 5000
```

