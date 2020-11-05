#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A collection of useful functions.
"""

import numpy as np

__all__ = ['DM']

def DM(distance):
	#distance in parsecs
	return 5.0*np.log10(distance)-5.0
