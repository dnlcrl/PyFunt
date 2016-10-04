from __future__ import division, print_function, absolute_import
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('im2col_cyt', ['im2col_cyt.pyx'],
    include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)



def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pyfunt', parent_package, top_path)
    config.make_config_py()
    return config

# if __name__ == '__main__':
#     from numpy.distutils.core import setup
#     setup(**configuration(top_path='').todict())
