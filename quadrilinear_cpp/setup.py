from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if torch.cuda.is_available():
# if False:
    print('Including CUDA code.')
    setup(
        name='quadrilinear4d',
        ext_modules=[
            CUDAExtension('quadrilinear4d', [
                'src/quadrilinear4d_cuda.cpp',
                'src/quadrilinear4d_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found. Fall back to CPU.')
    setup(name='quadrilinear4d',
        ext_modules=[CppExtension(name = 'quadrilinear4d', 
                                  sources= ['src/quadrilinear4d.cpp'],
                                  extra_compile_args=['-fopenmp'])],
        cmdclass={'build_ext': BuildExtension})
