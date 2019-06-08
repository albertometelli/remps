#!/usr/bin/env bash

# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

python ./experiments/run_experiment.py  --no-restore-variables \
                                            --no-save-variables \
                                            --hidden-layer-size 100 \
                                            --iteration-number 10000 \
                                            --env-id 2 \
                                            --initial-port 1000 \
                                            --n-trajectories 100000 \
                                            --max-steps 10000 \
                                            --eval-freq 2 \
                                            --eval-steps 2 \
                                            --kappa 1e-3 \
                                            --file-suffix "" \
                                            --no-normalize-data \
                                            --training-set-size 500 \
                                            --dual-reg 0 \
                                            --policy-reg 0