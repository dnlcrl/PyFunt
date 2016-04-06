from setuptools import setup

with open('README.md') as README:
    long_description = README.read()
    long_description = long_description[long_description.index('Description'):]

setup(name='pyfunt',
      version='0.1',
      description='Pythonic Deep Learning Framework',
      long_description=long_description,
      install_requires=['suds'],
      url='http://github.com/dnlcrl/PyFunt',
      author='Daniele E. Ciriello',
      author_email='ciriello.daniele@gmail.com',
      license='MIT',
      packages=['pyfunt'],
      scripts=['scripts/pyfunt'],
      keywords='pyfunt deep learning artificial neural network convolution'
      )
