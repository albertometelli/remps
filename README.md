# CMDP
Repository for the paper "Reinforcement Learning in Configurable Continuous Environments". Accepted at ICML 2019.

# Introduction
A CMDP is a MDP in which the transition function is affected by some configurable parameters.
The effect of the parameter on the transition function is unknown.

# Documentation
The class cmdp is the abstract class of the environment providing the method setParams for the setting of the parameters.
An environment should implement this class.

# Installation
Install dependencies inside requirements.txt
Using setup.py
```
cd cmdp
pip install -e .
```

# Run
## REMPS on Mountain Car

Mountain car has env_id = 0.
Omega is the initial value of the environment parameter.
```
python ${ROOT}/cmdp/runExperiment.py --no-render \
                                        --train-model-policy \
                                        --no-restore-variables \
                                        --no-save-variables \
                                        --hidden-layer-size 0 \
                                        --iteration-number 10000 \
                                        --omega 8 \
                                        --reward-type 3 \
                                        --env-id 0 \
                                        --n-actions 2 \
                                        --n-trajectories 100 \
                                        --max-steps 1000 \
                                        --eval-freq 2 \
                                        --eval-steps 2 \
                                        --noise-std 1e-5  \
                                        --epsilon 2 \
                                        --use-remps \
```

## REMPS on CartPole
Cartpole has env_id=1. Noise_std is useless in cartpole.
Omega is the initial value of the environment parameter.
```
python ${ROOT}/cmdp/runExperiment.py --no-render \
                                        --train-model-policy \
                                        --no-restore-variables \
                                        --no-save-variables \
                                        --hidden-layer-size 0 \
                                        --iteration-number 10000 \
                                        --omega 8 \
                                        --reward-type 3 \
                                        --env-id 1 \
                                        --n-actions 2 \
                                        --n-trajectories 100 \
                                        --max-steps 1000 \
                                        --eval-freq 2 \
                                        --eval-steps 2 \
                                        --noise-std 1e-5  \
                                        --epsilon 2 \
                                        --use-remps \
```

## REMPS on Chain
Add the epsilons you are interested in in runChain.
```
python runChain.py  --max-steps 500 \
                    --n_trajectories 10
```


# Author
Emanuele Ghelfi

Supervisors:
- Alberto Maria Metelli
- Marcello Restelli
