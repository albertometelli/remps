#!/usr/bin/env bash

# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

python ./cmdp/rungpomdp.py --no-render \
                                     --train-model-policy \
                                     --no-restore-variables \
                                     --no-save-variables \
                                     --hidden-layer-size 0 \
                                     --iteration-number 2000  \
                                     --omega 8 \
                                     --reward-type 3 \
                                     --env-id 1 \
                                     --n-actions 2 \
                                     --n-trajectories 250 \
                                     --max-steps 100 \
                                     --eval-freq 2 \
                                     --eval-steps 2 \
                                     --noise-std 1e-5 \
                                     --epsilon 0.0001 \
                                     --use-remps \
                                     --file-suffix 'NN' \
                                     --normalize-data \
                                     --training-set-size 100