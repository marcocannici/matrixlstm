#!/usr/bin/env bash
# .sh GPU CMD
gpu=$1

if [ "$#" -eq 0 ]; then
    echo "USAGE script.sh GPU -c <config_path> --density <density> --output_file <out_csv_path>"
    return
fi
shift 1

chmod +x entrypoint.sh
NV_GPU=$gpu nvidia-docker run -it --rm \
     --runtime=nvidia \
     --shm-size=8G \
     -v ${PWD}:/exp \
     --user $(id -u):$(id -g) \
     -e PYTHONPATH=/exp \
     -e LOGNAME=${USER} \
     --entrypoint="./entrypoint.sh" \
     -t matrixlstm/classification \
     python scripts/profile_gpu_memory.py ${@}
