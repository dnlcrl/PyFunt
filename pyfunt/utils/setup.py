from __future__ import division, print_function, absolute_import
from distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('utils', parent_package, top_path)
    config.make_config_py()
    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
