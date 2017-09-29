#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains class to generate all predicted model parameters from sample of 
stellar parameters.
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator


# define aliases for the MIST EEP tracks
currentpath = __file__
if currentpath[-1] == 'c':
	removeind = -27
else:
	removeind = -26
MISTFILE_DEFAULT = os.path.dirname(__file__[:removeind]+'data/MIST/')

class MISTgen(object):
	"""
	The basic class to geneate predicted parameters from the MIST models 
	from a set of EEP/initial_mass/initial_[Fe/H]/initial_[a/Fe] 
	when using isochrones, (or in the future 
	Teff/log(g)/[Fe/H]/[a/Fe] parameters from the Payne).


	"""
	def __init__(self, model=None,verbose=True):
		super(MISTgen, self).__init__()

		# check for a user defined model
		if model != None:
			self.mistfile = model
		else:
			self.mistfile = MISTFILE_DEFAULT+'MIST_1.1_EEPtrk.h5'

		self.verbose = verbose
		if self.verbose:
			print('Using Model: {0}'.format(self.mistfile))

		# read in HDF5 model
		misth5 = h5py.File(self.mistfile,'r')

		# build MIST object to eventually stick into interpolator

		# the array of parameters stored in object. Not full eep track
		# due to memory concerns

		inpararr = ([
			'initial_mass','initial_[Fe/H]','initial_[a/Fe]',
			'log_age','star_mass','log_R','log_L'
			'log_Teff','[Fe/H]','log_g',
			])

		for kk in mist5h['index']:
			# read in MIST array
			mist_i = np.array(misth5[kk])
			mist_i = mist_i[pararr]

			# create EEP array inside mist_i
			mist_i = np.lib.recfunctions.append_fields(mist_i,'EEP',range(1,len(mist_i)+1))

			if kk == mist5h['index'][0]:
				self.mist = mist_i.copy()
			else:
				self.mist = np.concatenate([self.mist,mist_i])

		# rename the fields to better column names
		self.mist = np.lib.recfunctions.rename_fields(self.mist,
			{
			'log_age':'log(Age)',
			'star_mass':'Mass',
			'log_R':'log(Rad)',
			'log_L':'log(L)',
			'log_Teff':'log(Teff)',
			'log_g':'log(g)',
			})
		
		# calculate min and max values for each of the sampled arrays
		self.minmax = {}
		self.minmax['EEP']  = [self.mist['EEP'].min(), self.mist['EEP'].max()]
		self.minmax['MASS'] = [self.mist['initial_mass'].min(), self.mist['initial_mass'].max()]
		self.minmax['FEH']  = [self.mist['initial_[Fe/H]'].min(), self.mist['initial_[Fe/H]'].max()]

		# build KD-Tree
		if self.verbose:
			print 'Growing the KD-Tree...'
		# determine unique values in grid
		self.eep_uval  = np.unique(self.mist['EEP'])
		self.mass_uval = np.unique(self.mist['initial_mass'])
		self.feh_uval  = np.unique(self.mist['initial_[Fe/H]'])

		# determine difference between unique values
		self.eep_diff  = np.diff(self.eep_uval)
		self.mass_diff = np.diff(self.mass_uval)
		self.feh_diff  = np.diff(self.feh_uval)

		# determine which unique grid point each EEP point belongs to
		self.eep_dig  = np.digitize(self.mist['EEP'],bins=self.eep_uval,right=True)
		self.mass_dig = np.digitize(self.mist['initial_mass'],bins=self.mass_uval,right=True)
		self.feh_dig  = np.digitize(self.mist['initial_[Fe/H]'],bins=self.feh_uval,right=True)

		# combine digitized arrays
		self.pts_n = np.array([self.eep_dig,self.mass_dig,self.feh_dig]).T

		# build tree and set distance metric (2-pts)
		self.tree = cKDTree(self.pts_n)
		self.dist = np.sqrt( (2.0**2.0) + (2.0**2.0) + (2.0**2.0) )

		# create stacked array for interpolation
		self.valuestack = np.stack(
			[self.mist[kk] for kk in self.mist.dtype.names],
			axis=1)

	def getMIST(self,pars,**kwargs):
		# unbind the input pars
		eep,mass,feh = pars

		# check to make sure pars are within bounds of EEP tracks
		if ((eep  > self.minmax['EEP'][1]) or 
			(eep  < self.minmax['EEP'][0]) or 
			(mass > self.minmax['MASS'][1]) or 
			(mass < self.minmax['MASS'][0]) or 
			(feh  > self.minmax['FEH'][1]) or
			(feh  < self.minmax['FEH'][0])
			):
			if verbose:
				print 'HIT MODEL BOUNDS'
			return 'ValueError'

		# build output dictionary to handle everything
		moddict = {}

		# stick in input pars
		moddict['EEP'] = eep
		moddict['Mass'] = mass
		moddict['[Fe/H]in'] = feh

		ind_eep  = np.digitize(eep,self.eep_dig,right=False)-1
		ind_mass = np.digitize(mass,self.mass_dig,right=False)-1
		ind_feh  = np.digitize(feh,self.feh_dig,right=False)-1

		inds = np.array([ind_eep,ind_mass,ind_feh])
		inds = np.squeeze(inds)

		find_eep  = (eep-self.eep_dig[ind[0]])/self.eep_diff[ind[1]]
		find_mass = (mass-self.mass_dig[ind[1]])/self.mass_diff[ind[1]]
		find_feh  = (feh-self.feh_dig[ind[2]])/self.feh_diff[ind[1]]

		KDTind = self.tree.query_ball_point([find_eep,find_mass,find_feh],self.dist,p=5)
		KDTpars = self.mist[KDTind]

		return KDTpars