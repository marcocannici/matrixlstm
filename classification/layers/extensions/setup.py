from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
      name='matrixlstm_helpers',
      ext_modules=[
            CUDAExtension('matrixlstm_helpers', [
                  'matrixlstm_helpers.cpp',
                  'matrixlstm_helpers_kernel.cu',
            ],
                          # extra_compile_args={
                          #       'cxx': ['-g'],
                          #       'nvcc': ['-arch=compute_20']})
                          )],
      cmdclass={
            'build_ext': BuildExtension
      })
