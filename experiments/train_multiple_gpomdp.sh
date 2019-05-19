#!/usr/bin/env bash

# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

for s in 10 1 1000 57 2 3 4876 907655 90 46; do
    echo ${s}
    COMMAND="source /anaconda3/bin/activate remps; python ./remps/rungpomdp.py --no-render \
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
                                     --training-set-size 100 \
                                     --seed "
    echo $COMMAND$s
    echo ${COMMAND}${s}

    command_all=${COMMAND}${s}

    echo ${command_all}

    screen -S gpomdp${s} -d -m bash -c "source activate remps; python ./remps/rungpomdp.py --no-render \
                                     --train-model-policy \
                                     --no-restore-variables \
                                     --no-save-variables \
                                     --hidden-layer-size 0 \
                                     --iteration-number 100  \
                                     --omega 8 \
                                     --reward-type 3 \
                                     --env-id 2 \
                                     --n-actions 2 \
                                     --n-trajectories 40 \
                                     --max-steps 500 \
                                     --eval-freq 2 \
                                     --eval-steps 2 \
                                     --noise-std 1e-5 \
                                     --epsilon 0.0001 \
                                     --use-remps \
                                     --normalize-data \
				     --file-suffix 'random-init' \
                                     --exact \
                                     --training-set-size 100 \
                                     --seed "${s}
done
