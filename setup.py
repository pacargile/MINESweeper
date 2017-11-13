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
    name="MINESweeper_V2",
    url="https://github.com/pacargile/MINESweeper_V2.0",
    version="2.0",
    author="Phillip Cargile",
    author_email="pcargile@cfa.harvard.edu",
    packages=["MINESweeper_V2"],
    license="LICENSE",
    description="MIST Isochrones w/ Nested Sampling",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "dynesty"],
)

