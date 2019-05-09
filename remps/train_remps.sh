#!/usr/bin/env bash
# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

python ./remps/runExperiment.py --no-render \
                                            --train-model-policy \
                                            --no-restore-variables \
                                            --no-save-variables \
                                            --hidden-layer-size 0 \
                                            --iteration-number 2000 \
                                            --omega 8 \
                                            --reward-type 3 \
                                            --env-id 3 \
                                            --n-actions 2 \
                                            --n-trajectories 10000 \
                                            --max-steps 500 \
                                            --eval-freq 2 \
                                            --eval-steps 2 \
                                            --noise-std 1e-5  \
                                            --epsilon 1e-3 \
                                            --use-remps \
                                            --file-suffix "exact-random-init" \
                                            --no-normalize-data \
                                            --training-set-size 500 \
                                            --dual-reg 0 \
                                            --policy-reg 0
