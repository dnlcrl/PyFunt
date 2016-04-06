#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

with open('README.md') as README:
    long_description = README.read()
    long_description = long_description[long_description.index('Description'):]

extensions = [
    Extension('pyfunt/layers/im2col_cython', ['pyfunt/layers/im2col_cython.pyx'],
              include_dirs=[numpy.get_include()],
              build_ext=['-b', 'pyfunt/layers']
              ),
]

setup(name='pyfunt',
      version='0.1',
      description='Pythonic Deep Learning Framework',
      long_description=long_description,
      install_requires=['Cython==0.23.4',
                        'matplotlib==1.5.1',
                        'numpy==1.10.4',
                        'scipy==0.17.0',
                        'scikit_image==0.11.3',
                        'tqdm==3.8.0',
                        'cv2==1.0',
                        'scikit_learn == 0.17.1'],
      url='http://github.com/dnlcrl/PyFunt',
      author='Daniele E. Ciriello',
      author_email='ciriello.daniele@gmail.com',
      license='MIT',
      packages=['pyfunt'],
       ext_modules = cythonize("pyfunt/layers/*.pyx"),
      keywords='pyfunt deep learning artificial neural network convolution'
      )

#os.rename('im2col_cython.so', 'pyfunt/layers/im2col_cython.so') # :'( fixme
