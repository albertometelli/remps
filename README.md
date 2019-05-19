# REMPS
Repository for the paper "Reinforcement Learning in Configurable Continuous Environments". Accepted at ICML 2019.

# Introduction
A Conf-MDP is a MDP in which the transition function p: (s,a) -> s' is affected by some configurable parameters \omega.
The effect of the parameter on the transition function can be known (exact case) or unknown (approximated case).

# Documentation
The class cmdp is the abstract class of the environment providing the method setParams for the setting of the parameters.
An environment should implement this class.

# Installation
Install dependencies inside requirements.txt

```
pip install -r requirements.txt
```

Using setup.py:

```
cd remps
pip install -e .
```

# Run

## REMPS on CartPole
Cartpole has env_id=1. Noise_std is useless in cartpole.
Omega is the initial value of the environment parameter.
```
python ${ROOT}/remps/runExperiment.py --no-render \
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


# Authors
- Alberto Maria Metelli
- Emanuele Ghelfi
- Marcello Restelli
