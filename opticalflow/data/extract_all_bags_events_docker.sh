#!/usr/bin/env bash
chmod +x extract_all_bags_events.sh
docker run -it --rm \
   -v ${PWD}:/exp \
   --user $(id -u):$(id -g) \
   -w /exp \
   -t matrixlstm/opticalflow-data \
   ./extract_all_bags_events.sh