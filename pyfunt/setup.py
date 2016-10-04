from __future__ import division, print_function, absolute_import
from distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('pyfunt', parent_package, top_path)
    config.add_subpackage('examples')
    config.add_extension('im2col_cyt',
                         sources=[('im2col_cyt.c')],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
