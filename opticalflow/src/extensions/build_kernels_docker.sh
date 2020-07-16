docker run -it --rm \
   -v ${PWD}:/exp \
   -w /exp \
   -t tensorflow/tensorflow:1.12.0-devel-gpu \
   sh build_kernels.sh