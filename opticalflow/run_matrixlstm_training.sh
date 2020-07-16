#!/usr/bin/env bash
# .sh GPU -c CONFIG_PATH
gpu=$1

if [ "$#" -eq 0 ]; then
    echo "USAGE script.sh GPU -c CONFIG_PATH [ARGS]"
    return
fi
shift 1

NV_GPU=$gpu nvidia-docker run -it --rm \
     --runtime=nvidia \
     --shm-size=8G \
     -v ${PWD}:/exp \
     --user $(id -u):$(id -g) \
     -e PYTHONPATH=/exp \
     -t matrixlstm/opticalflow-experiments \
     python src/train_matrixlstm.py ${@}
