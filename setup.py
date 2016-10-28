from setuptools import setup
import numpy as np
from Cython.Build import cythonize

setup(
    name='svae',
    author='Matthew James Johnson and David Duvenaud',
    author_email='mattjj@csail.mit.edu, duvenaud@cs.toronto.edu',
    packages=['svae', 'autograd_linalg'],
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),],
    install_requires=['autograd >= 1.1.7', 'numpy >= 1.10.0', 'cython >= 0.24',
                      'scipy >= 0.18.1'],
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 2.7'],
)
