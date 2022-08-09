#!/usr/bin/env python

# Copyright (C) 2022 Anatoli Ulmer

from setuptools import setup, find_packages
#from Cython.Build import cythonize
import numpy
import os

import versioneer

setup(name='lv17analysis',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      include_package_data = True,
      author='Anatoli Ulmer',
      author_email='anatoli.ulmer@gmail.com',
      description='Python package for data analysis of tmolv1720 experiment (2022).',
      url='https://github.com/skuschel/tmolv1720',
      install_requires=['matplotlib>=1.3',
                        # ndarray.tobytes was introduced in np 1.9
                        'numpy>=1.9', 'numpy>=1.9',
                        'scipy', 'future', 'urllib3', 'numexpr', 'cython>=0.18']
      )
