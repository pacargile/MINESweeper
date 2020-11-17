#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

    
setup(
    name="MINESweeper",
    url="https://github.com/pacargile/MINESweeper.git",
    version="2.0",
    author="Phillip Cargile",
    author_email="pcargile@cfa.harvard.edu",
    packages=["minesweeper",
              "minesweeper.jax",],
    license="LICENSE",
    description="MIST Isochrones w/ Nested Sampling",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "dynesty", "torch"],
)

# write top level __init__.py file with the correct absolute path to package repo
toplevelstr = ("""try:
    from ._version import __version__
except(ImportError):
    pass

"""
)

with open('minesweeper/__init__.py','w') as ff:
  ff.write(toplevelstr)
  ff.write('\n')
  ff.write("""__abspath__ = '{0}/'\n""".format(os.getcwd()))
