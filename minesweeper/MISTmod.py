#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains class to generate all predicted model parameters from sample of 
stellar parameters.
"""

import os
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import h5py
from scipy.spatial import KDTree as KDTree
from scipy.interpolate import LinearNDInterpolator

from numpy.lib import recfunctions
import minesweeper

# # define aliases for the MIST EEP tracks
# currentpath = __file__
# if currentpath[-1] == 'c':
#   removeind = -27
# else:
#   removeind = -26
# MISTFILE_DEFAULT = os.path.dirname(__file__[:removeind]+'data/MIST/')

# the array of parameters stored in object. Not full eep track
# due to memory concerns

# inpararr = ([
#   'EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]',
#   'log_age','star_mass','log_R','log_L',
#   'log_Teff','[Fe/H]','log_g',
#   ])
# renameparr = ([
#   'EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]',
#   'log(Age)','Mass','log(Rad)','log(L)',
#   'log(Teff)','[Fe/H]','log(g)',
#   ])

# dictionary to translate par names to MIST names
MISTrename = {
    'log(Age)':'log_age',
    'Mass':'star_mass',
    'log(R)':'log_R',
    'log(L)':'log_L',
    'log(Teff)':'log_Teff',
    'log(g)':'log_g',
}

class GenMIST(object):
    """
    The basic class to geneate predicted parameters from the MIST models 
    from a set of EEP/initial_mass/initial_[Fe/H]/initial_[a/Fe] 
    when using isochrones, (or in the future 
    Teff/log(g)/[Fe/H]/[a/Fe] parameters from the Payne).


    """
    def __init__(self,**kwargs):
        super(GenMIST, self).__init__()

        # check for a user defined model
        self.mistfile = kwargs.get('model',None)

        mistfile = kwargs.get('MISTpath',None)
        if mistfile is None:
            self.mistfile = minesweeper.__abspath__+'data/MIST/MIST_2.0_EEPtrk.h5'
            # self.mistfile = 
        else:
            self.mistfile = mistfile
        # else:
        #   # define aliases for the MIST isochrones and C3K/CKC files
        #   currentpath = __file__
        #   if currentpath[-1] == 'c':
        #       removeind = -27
        #   else:
        #       removeind = -26
        #   self.MISTpath = os.path.dirname(__file__[:removeind]+'data/MIST/')

        #   self.mistfile = self.MISTpath+'/MIST_1.2_EEPtrk.h5'

        self.verbose = kwargs.get('verbose',True)

        # turn on age weighting
        self.ageweight = kwargs.get('ageweight',True)

        # list of values you want to interpolate over
        self.labels = kwargs.get('labels',['EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]'])

        # list of output parametrs you want from MIST 
        # in addition to EEP, init_mass, init_FeH
        self.predictions = kwargs.get('predictions',
            ['log(Age)','Mass','log(R)','log(L)',
            'log(Teff)','[Fe/H]','[a/Fe]','log(g)'])
        if type(self.predictions) == type(None):
            self.predictions = (['log(Age)','Mass','log(R)',
                'log(L)','log(Teff)','[Fe/H]','[a/Fe]','log(g)'])

        if self.verbose:
            print('Using Model: {0}'.format(self.mistfile))

        # read in HDF5 model
        misth5 = h5py.File(self.mistfile,'r')

        # build MIST object to eventually stick into interpolator
        self.modpararr = self.labels+self.predictions
        trans_modpararr = [MISTrename[x] if x in MISTrename.keys() else x for x in self.modpararr]
        for kk in misth5['index']:
            # read in MIST array
            mist_i = np.array(misth5[kk])
            mist_i = mist_i[trans_modpararr]

            if kk == misth5['index'][0]:
                self.mist = mist_i.copy()
            else:
                self.mist = np.concatenate([self.mist,mist_i])

        if self.ageweight:
            # print('... Fitting w/ equal Age weighting')
            self.predictions.append('Agewgt')
            self.modpararr.append('Agewgt')

        # build KD-Tree
        if self.verbose:
            print('Growing the KD-Tree...')
        # determine unique values in grid
        self.eep_uval  = np.unique(self.mist['EEP'])
        self.mass_uval = np.unique(self.mist['initial_mass'])
        self.feh_uval  = np.unique(self.mist['initial_[Fe/H]'])
        self.afe_uval  = np.unique(self.mist['initial_[a/Fe]'])

        # calculate min and max values for each of the sampled arrays
        self.minmax = {}
        self.minmax['EEP']  = [self.eep_uval.min(), self.eep_uval.max()]
        self.minmax['MASS'] = [self.mass_uval.min(),self.mass_uval.max()]
        self.minmax['FEH']  = [self.feh_uval.min(),self.feh_uval.max()]
        self.minmax['AFE']  = [self.afe_uval.min(),self.afe_uval.max()]

        # determine difference between unique values
        self.eep_diff  = np.diff(self.eep_uval)
        self.mass_diff = np.diff(self.mass_uval)
        self.feh_diff  = np.diff(self.feh_uval)
        self.afe_diff  = np.diff(self.afe_uval)

        # determine which unique grid point each EEP point belongs to
        self.eep_dig  = np.digitize(self.mist['EEP'],bins=self.eep_uval,right=True)
        self.mass_dig = np.digitize(self.mist['initial_mass'],bins=self.mass_uval,right=True)
        self.feh_dig  = np.digitize(self.mist['initial_[Fe/H]'],bins=self.feh_uval,right=True)
        self.afe_dig  = np.digitize(self.mist['initial_[a/Fe]'],bins=self.afe_uval,right=True)

        # combine digitized arrays
        self.pts_n = np.array([self.eep_dig,self.mass_dig,self.feh_dig,self.afe_dig]).T

        # build tree and set distance metric (2-pts)
        self.tree = KDTree(self.pts_n)
        searchrad = 2.0
        self.dist = np.sqrt( 
            (searchrad**2.0) + 
            (searchrad**2.0) + 
            (searchrad**2.0) + 
            (searchrad**2.0))

        # create stacked array for interpolation
        cols = [MISTrename[x] if x in MISTrename.keys() else x for x in self.predictions]
        self.valuestack = np.stack(
            [self.mist[kk] for kk in cols],
            axis=1)


    def getMIST(self,eep=300,mass=1.0,feh=0.0,afe=0.0,**kwargs):
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = True

        # unbind the input pars
        # eep = pars[0]
        # mass = pars[1]
        # feh = pars[2]

        # check to make sure pars are within bounds of EEP tracks
        if ((eep  > self.minmax['EEP'][1]) or 
            (eep  < self.minmax['EEP'][0]) or 
            (mass > self.minmax['MASS'][1]) or 
            (mass < self.minmax['MASS'][0]) or 
            (feh  > self.minmax['FEH'][1]) or
            (feh  < self.minmax['FEH'][0]) or
            (afe  > self.minmax['AFE'][1]) or
            (afe  < self.minmax['AFE'][0])
            ):
            if verbose:
                print('HIT MODEL BOUNDS')
            return None

        # build output dictionary to handle everything
        outpred = []

        # stick in input pars
        outpred.append(eep)
        outpred.append(mass)
        outpred.append(feh)
        outpred.append(afe)

        # # check to see if user is passing Dist and Av, if so stick it into moddict as well
        # if len(pars) > 3:
        #   moddict['Dist'] = pars[3]
        #   moddict['Av'] = pars[4]

        ind_eep  = np.digitize(eep,bins=self.eep_uval,right=False)-1
        ind_mass = np.digitize(mass,bins=self.mass_uval,right=False)-1
        ind_feh  = np.digitize(feh,bins=self.feh_uval,right=False)-1
        ind_afe  = np.digitize(afe,bins=self.afe_uval,right=False)-1

        find_eep  = ((eep-self.eep_uval[ind_eep])/self.eep_diff[ind_eep]) + ind_eep
        find_mass = ((mass-self.mass_uval[ind_mass])/self.mass_diff[ind_mass]) + ind_mass
        find_feh  = ((feh-self.feh_uval[ind_feh])/self.feh_diff[ind_feh]) + ind_feh
        find_afe  = ((afe-self.afe_uval[ind_afe])/self.afe_diff[ind_afe]) + ind_afe

        KDTind = self.tree.query_ball_point(
            [find_eep,find_mass,find_feh,find_afe],
            self.dist,
            p=1,eps=3)

        # pull from the value stack
        valuestack_i = self.valuestack[KDTind]

        # pull from the MIST isochrones
        KDTpars = self.mist[KDTind]

        # check to make sure there are at least two points in each dimension
        for testkk in ['EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]']:
            if len(np.unique(KDTpars[testkk])) < 2:
                if verbose:
                    print('Not enough points in KD-Tree sample')
                return None

        # do the linear N-D interpolation
        try:
            dataint = LinearNDInterpolator(
                    (KDTpars['EEP'],KDTpars['initial_mass'],KDTpars['initial_[Fe/H]'],KDTpars['initial_[a/Fe]']),
                    valuestack_i,
                    fill_value=np.nan_to_num(-np.inf),
                    rescale=True
                    )(eep,mass,feh,afe)
        except:
            if verbose:
                print('Problem with linear inter of KD-Tree sample')
                print(min(KDTpars['EEP']),max(KDTpars['EEP']),
                    min(KDTpars['initial_mass']),max(KDTpars['initial_mass']),
                    min(KDTpars['initial_[Fe/H]']),max(KDTpars['initial_[Fe/H]']),
                    min(KDTpars['initial_[a/Fe]']),max(KDTpars['initial_[a/Fe]'])
                    )
                print (eep,lage,feh,afe)
            return None

        # stick interpolated pars into outpred list
        for dd in dataint:
            if dd != np.nan_to_num(-np.inf):
                outpred.append(dd)
            else:
                if verbose:
                    print('Tried to extrapolate')
                return None

        # # stick interpolated pars into moddict
        # for ii,pp in enumerate(KDTpars.dtype.names):
        #   if dataint[ii] != np.nan_to_num(-np.inf):
        #       if pp not in ['EEP','initial_mass','initial_[Fe/H]']:
        #           moddict[pp] = dataint[ii]
        #   else:
        #       if verbose:
        #           print('Tried to extrapolate')
        #       return 'ValueError'

        return outpred