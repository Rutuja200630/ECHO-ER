from setuptools import setup, Extension
import pybind11
import numpy as np

ext_modules = [
    Extension(
        'fast_topk',
        ['src/fast_topk.cpp'],
        include_dirs=[pybind11.get_include(), np.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-ffast-math', '-march=native', '-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
]

setup(
    name='fast_topk',
    version='0.1.0',
    author='ECHO-ER',
    description='Fast Top-K extraction for sparse matrices',
    ext_modules=ext_modules,
)
