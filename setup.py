#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='Guerilla',
      version='1.0.2',
      description='Deep Learning Chess Engine',
      license="MIT",
      maintainer='Stephane Aroca-Ouellette',
      maintainer_email='stephanearocaouellette@gmail.com',
      author='Miguel Aroca-Ouellette',
      author_email='mig_ao@live.com',
      url='https://www.python.org/sigs/distutils-sig/',
      install_requires=['numpy', 'ruamel.yaml<=0.15', 'python-chess'],
      packages=['guerilla', 'guerilla.play'],
      package_data={'guerilla': ['data/weights/default.p', 'data/hyper_params/neural_net/default.yaml']},
      scripts=['scripts/Guerilla']
)