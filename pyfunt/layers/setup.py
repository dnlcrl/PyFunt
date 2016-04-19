#!/usr/bin/env python
from __future__ import division, print_function, absolute_import


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('layers', parent_package, top_path)

    config.add_extension('im2col_cython',
                         sources=[('im2col_cython.c')],
                         include_dirs=[get_numpy_include_dirs()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy

# extensions = [
#     Extension('im2col_cython', ['im2col_cython.pyx'],
#               include_dirs=[numpy.get_include()]
#               ),
# ]

# setup(
#     ext_modules=cythonize(extensions),
# )
