#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The top-level code to initialize and run the MINESweeper fitter
"""

        def _initphotnn(self,filterarray):
                import nnBC

                self.nnBCdict = {}
                self.BClist = []                
                for ff in filterarray:
                        self.BClist.append(ff)
                        try:
                                self.nnBCdict[ff] = nnBC.nnBC(
                                        nnh5='/n/conroyfs1/pac/ThePayne/SED/nnMIST_{0}.h5'.format(ff))
                                        # nnh5='/Users/pcargile/Astro/MIST/nnMIST/nnMIST_{0}.h5'.format(ff))
                        except IOError:
                                print('Cannot find NN HDF5 file for {0}'.format(ff))
