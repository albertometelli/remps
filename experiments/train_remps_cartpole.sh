#!/usr/bin/env bash
# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

python ./experiments/run_experiment.py  --no-restore-variables \
                                            --no-save-variables \
                                            --hidden-layer-size 0 \
                                            --iteration-number 2000 \
                                            --omega 8 \
                                            --env-id 1 \
                                            --n-trajectories 100000 \
                                            --max-steps 200 \
                                            --eval-freq 2 \
                                            --eval-steps 2 \
                                            --kappa 1e-2 \
                                            --file-suffix "exact-fixed-init" \
                                            --no-normalize-data \
                                            --training-set-size 500 \
                                            --dual-reg 0 \
                                            --policy-reg 0