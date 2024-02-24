## Entropy-regularized Diffusion Policy with Q-Ensembles for Offline Reinforcement Learning


## Dependenices

* OS: Ubuntu 20.04
* nvidia :
	- cuda: 11.7
	- cudnn: 8.5.0
* python3
* pytorch >= 1.13.0

## How to run the code

```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --lr_decay
```

All the hyperparameters are fixed in the `main.py`.