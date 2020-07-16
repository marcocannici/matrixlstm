#!/bin/bash
cp /exp/* /tensorflow/tensorflow/core/user_ops && \
cd /tensorflow/ && \
bazel build --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:matrixlstm_helpers.so && \
cp /tensorflow/bazel-bin/tensorflow/core/user_ops/*.so /exp/ && \
#bazel build --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:zero_out_op_kernel_1.so && \
#cp /tensorflow/bazel-bin/tensorflow/core/user_ops/*.so /exp/ && \
cd /exp && \
/bin/sh -c "$*"


