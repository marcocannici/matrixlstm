import torch
from torch.utils.cpp_extension import load

# Compile extensions at runtime
matrixlstm_helpers = load(name="matrixlstm_helpers",
                          sources=["matrixlstm_helpers.cpp"])
