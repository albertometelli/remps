# REMPS
Repository for the paper "Reinforcement Learning in Configurable Continuous Environments". Accepted at ICML 2019.

# Introduction
A Conf-MDP is a MDP in which the transition function p: (s,a) -> s' is affected by some configurable parameters \omega.
The effect of the parameter on the transition function can be known (exact case) or unknown (approximated case).

# Documentation
The class `confMDP is the abstract class of the environment providing the method setParams for the setting of the parameters.
An environment should implement this class.

# Installation
Install dependencies inside requirements.txt

```bash
pip install -r requirements.txt
```

Using setup.py:

```bash
cd remps
pip install -e .
```

# Run

## REMPS on CartPole
Cartpole has env_id=1. Noise_std is useless in cartpole.
Omega is the initial value of the environment parameter.
```bash
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

```bash
python runChain.py  --max-steps 500 \
                    --n_trajectories 10
```

## REMPS on TORCS

Download the TORCS code:

```bash
git submodule update --init
```
The torcs code is inside the folder `libs/gym_torcs`.
Install torcs following the steps listed [here](https://github.com/ugo-nama-kun/gym_torcs/tree/master/vtorcs-RL-color)

Run the torcs experiment with:

```bash
./experiments/train_remps_torcs.sh
```

# Authors
- Alberto Maria Metelli
- Emanuele Ghelfi [emanuele.ghelfi@mail.polimi.it](emanuele.ghelfi@mail.polimit.it) [Scholar](https://scholar.google.it/citations?hl=it&view_op=list_works&gmla=AJsN-F5qKVISBHxU3To19s-Iq8xA1c3BivcIXYD1DKEvcky2mcdfiF3lMg4JjrmOE8fK1Jiikj9XfUyF54s8HnXJMYeBdpPLwaCJ8lMlVhHOy178vQAvGwA&user=JJqNoGQAAAAJ) [Github](https://github.com/EmanueleGhelfi) [Website](emanueleghelfi.github.io) [Twitter](twitter.com/manughelfi)
- Marcello Restelli


# Citing

```
@InProceedings{pmlr-v97-metelli19a,
  title = 	 {Reinforcement Learning in Configurable Continuous Environments},
  author = 	 {Metelli, Alberto Maria and Ghelfi, Emanuele and Restelli, Marcello},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {4546--4555},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},`
  pdf = 	 {http://proceedings.mlr.press/v97/metelli19a/metelli19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/metelli19a.html},
  abstract = 	 {Configurable Markov Decision Processes (Conf-MDPs) have been recently introduced as an extension of the usual MDP model to account for the possibility of configuring the environment to improve the agent’s performance. Currently, there is still no suitable algorithm to solve the learning problem for real-world Conf-MDPs. In this paper, we fill this gap by proposing a trust-region method, Relative Entropy Model Policy Search (REMPS), able to learn both the policy and the MDP configuration in continuous domains without requiring the knowledge of the true model of the environment. After introducing our approach and providing a finite-sample analysis, we empirically evaluate REMPS on both benchmark and realistic environments by comparing our results with those of the gradient methods.}
}
``