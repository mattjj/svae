from setuptools import setup
import numpy as np
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),],
)
