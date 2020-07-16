#!/usr/bin/env bash
# .sh GPU CHECKPOINT_PATH -c CONFIG_PATH
load_path=/exp/data/log/saver/
gpu=$1
trained_model=$2
gt_path=/exp/data/download/indoor_flying/

if [ "$#" -eq 0 ]; then
    echo "USAGE script.sh GPU MODEL-NAME -c CONFIG_PATH [ARGS]"
    return
fi
shift 2

for seqname in indoor_flying1_events indoor_flying2_events indoor_flying3_events;
do
    NV_GPU=$gpu nvidia-docker run -it --rm \
     --runtime=nvidia \
     --shm-size=8G \
     -v ${PWD}:/exp \
     --user $(id -u):$(id -g) \
     -e PYTHONPATH=/exp \
     -t matrixlstm/opticalflow-experiments \
     python src/test_matrixlstm.py --test_sequence $seqname --gt_path ${gt_path}${seqname}_gt_flow_dist.npz --load_path ${load_path} --training_instance ${trained_model} ${@}
done
