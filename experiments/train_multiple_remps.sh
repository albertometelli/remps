#!/usr/bin/env bash
# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`
 
for s in 10 1 1000 57 2 3 4876 907655 90 46; do
        echo ${s}
        screen -S remps${s} -d -m bash -c 'source activate remps && taskset -c 0-22 python ./remps/run_experiment.py --no-render \
                                                --no-restore-variables \
                                                --no-save-variables \
                                                --hidden-layer-size 0 \
                                                --iteration-number 100 \
                                                --omega 0.8 \
                                                --env-id 3 \
                                                --n-actions 2 \
                                                --n-trajectories 10000 \
                                                --max-steps 500 \
                                                --eval-freq 2 \
                                                --eval-steps 2 \
                                                --noise-std 1e-5  \
                                                --epsilon 0.01 \
						                        --file-suffix "random-init" \
                                                --no-normalize-data \
                                                --training-set-size 500 \
                                                --dual-reg 0 \
                                                --policy-reg 0 \
                                                --exact \
                                                --seed '${s}
done
