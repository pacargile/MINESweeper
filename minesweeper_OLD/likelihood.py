#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# dictionary to translate par names to MIST names
MISTrename = {
	'log_age':'log(Age)',
	'star_mass':'Mass',
	'log_R':'log(Rad)',
	'log_L':'log(L)',
	'log_Teff':'log(Teff)',
	'log_g':'log(g)',
}

class likelihood(object):
	"""docstring for likelihood"""
	def __init__(self,datadict,MISTinfo,**kwargs):
		super(likelihood, self).__init__()

		self.verbose = kwargs.get('verbose',True)

		ageweight = kwargs.get('ageweight',False)

		fastinterp = kwargs.get('fastinterp',True)

		self.predictions = kwargs.get('predictions',None)

		# the dictionary with all data for likelihood
		self.datadict = datadict

		# init the MIST models
		self.MISTgen = self._initMIST(
			model=MISTinfo['model'],stripeindex=MISTinfo['stripe'],
			ageweight=ageweight,fast=fastinterp,predictions=self.predictions)

		# if there is photometry, init the NN for those bands
		if 'filterarray' in MISTinfo.keys():
			self.ppsed = self._initSED(
				filterarray=MISTinfo['filterarray'],nnpath=MISTinfo['nnpath'])
		else:
			self.ppsed = None

	def _initSED(self,filterarray=None,nnpath=None, fast=True,**kwargs):

		if fast:
			from .predsed import FastPaynePredictor
			# init Payne Photometry module
			return FastPaynePredictor(usebands=filterarray,nnpath=nnpath)
		else:
			# OLD SLOWER ANN BASED ON PYTORCH
			from .predsed import PaynePredictor
			return PaynePredictor(usebands=filterarray,nnpath=nnpath)

	def _initMIST(self,model=None,ageweight=True,fast=True,predictions=None,**kwargs):

		if fast:
			print('... Using Fast MIST interpolation')
			from .fastMISTmod import fastMISTgen
			return fastMISTgen(model=model,predictions=predictions,
				ageweight=ageweight,verbose=self.verbose)
		else:
			print('... Using Scipy-based MIST interpolation')
			# OLD SLOWER ANN BASED ON SCIPY INTERPOLATE
			from .MISTmod import MISTgen
			# init MISTgen
			return MISTgen(
				model=model,predictions=predictions,
				ageweight=ageweight,verbose=self.verbose)

		
	def like(self,pars,returnarr=False):
		# split pars into MIST and [Dist,Av]
		eep = pars[0]
		mass = pars[1]
		FeH = pars[2]
		mistpars = [eep,mass,FeH]

		if len(pars) > 3:
			dist = pars[3]
			Av = pars[4]
			photpars = [dist,Av]
		else:
			photpars = None

		# run getMIST to pull model
		MIST_i_arr = self.MISTgen.getMIST(eep=eep,mass=mass,feh=FeH,verbose=False)

		if type(MIST_i_arr) == type(None):
			return -np.inf

		# stick MIST model pars into useful dict format
		self.MIST_i = ({
			kk:pp for kk,pp in zip(
				self.MISTgen.modpararr,MIST_i_arr)
			})			

		for kk in self.MIST_i.keys():
			if kk in MISTrename.keys():
				self.MIST_i[MISTrename[kk]] = self.MIST_i.pop(kk)

		if 'log(Teff)' in self.MIST_i.keys():
			self.MIST_i['Teff'] = 10.0**self.MIST_i['log(Teff)']
		if 'log(Rad)' in self.MIST_i.keys():
			self.MIST_i['Rad']  = 10.0**self.MIST_i['log(Rad)']

		if type(photpars) != type(None):
			self.MIST_i['Av'] = Av
			self.MIST_i['Dist'] = dist


		if returnarr:
			return self.calclikearray(self.MIST_i,photpars)
		else:
			return self.calclike(self.MIST_i,photpars)


	def calclike(self,modMIST,photpars):
		# create input arrays
		inpars = self.datadict.get('pars',None)
		inphot = self.datadict.get('phot',None)
		inspec = self.datadict.get('spec',None)

		# init lnlike
		lnlike = 0.0

		# set a default parallax
		parallax = None

		if 'Agewgt' in modMIST.keys():
			lnlike += np.log(modMIST['Agewgt'])

		# place a likelihood prob on star's age being within a 
		# rough estimate of a Hubble time ~16 Gyr
		if modMIST['log(Age)'] > 10.2:
			return -np.inf

		# check to see if there are any pars to fit
		if type(inpars) != type(None):
			for pp in inpars.keys():
				if pp in modMIST.keys():
					lnlike += -0.5 * ((inpars[pp][0]-modMIST[pp])**2.0)/(inpars[pp][1]**2.0)

				# check if parallax is given, if so store it for the photometric step
				if pp == 'Para':
					parallax = inpars['Para']

		# check to see if dist and Av are passed, if so fit the photometry
		if type(photpars) != type(None):
			dist = photpars[0]
			Av = photpars[1]

			# check for unphysical distance and reddening
			if dist <= 0.0:
				return -np.inf
			if Av <= 0.0:
				return -np.inf

			# fit a parallax if given
			if type(parallax) != type(None):
				parallax_i = 1000.0/modMIST['Dist']
				lnlike += -0.5 * ((parallax[0]-parallax_i)**2.0)/(parallax[1]**2.0)

			if type(inphot) != type(None):

				# create parameter dictionary
				photpars = {}
				photpars['logt'] = modMIST['log(Teff)']
				photpars['logg'] = modMIST['log(g)']
				photpars['feh']  = modMIST['[Fe/H]']
				photpars['logl'] = modMIST['log(L)']
				photpars['av']   = modMIST['Av']
				photpars['dist'] = modMIST['Dist']

				# create filter list and arrange photometry to this list
				filterlist = inphot.keys()
				inphotlist = [inphot[x] for x in filterlist]

				# sed = self.ppsed.sed(filters=filterlist,**photpars)
				sed = self.ppsed.sed(**photpars)

				for sed_i,inphot_i in zip(sed,inphotlist):
					lnlike += -0.5 * ((sed_i-inphot_i[0])**2.0)/(inphot_i[1]**2.0)

		return lnlike

	def calclikearray(self,modMIST,photpars):
		# define a likelihood function that returns an array of -0.5*chi-square values for 
		# input data. Useful when using optimizers.

		# create input arrays
		inpars = self.datadict.get('pars',None)
		inphot = self.datadict.get('phot',None)
		inspec = self.datadict.get('spec',None)

		lnlike = []
		parallax = None

		# have to define the keys in these dictionaries so that the order is standardized
		if type(inpars) != type(None):
			inpars_keys = inpars.keys()
			inpars_keys.sort()
		else:
			inpars_keys = []

		if type(inphot) != type(None):
			inphot_keys = inphot.keys()
			inphot_keys.sort()
		else:
			inphot_keys = []

		# check to see if there are any pars to fit
		if type(inpars) != type(None):
			for pp in inpars_keys:
				if pp in modMIST.keys():
					# lnlike.append(-0.5 * ((inpars[pp][0]-modMIST[pp])**2.0)/(inpars[pp][1]**2.0))
					lnlike.append(((inpars[pp][0]-modMIST[pp])/inpars[pp][1])**2.0)

				# check if parallax is given, if so store it for the photometric step
				if pp == 'Para':
					parallax = inpars['Para']

		# check to see if dist and Av are passed, if so fit the photometry
		if type(photpars) != type(None):
			dist = photpars[0]
			Av = photpars[1]

			# fit a parallax if given
			if type(parallax) != type(None):
				parallax_i = 1000.0/dist
				# lnlike.append(-0.5 * ((parallax[0]-parallax_i)**2.0)/(parallax[1]**2.0))
				lnlike.append(((parallax[0]-parallax_i)/parallax[1])**2.0)

			if type(inphot) != type(None):

				# define some parameters from modMIST for the photomteric predictions
				Teff = 10.0**modMIST['log(Teff)']
				FeH  = modMIST['[Fe/H]']
				logg = modMIST['log(g)']

				# calc bolometric magnitude
				Mbol = -2.5*modMIST['log(L)']+4.74

				# for each filter, calculate predicted mag and calc likelihood
				for kk in inphot_keys:
					BC_i = float(self.ANNfn[kk].eval([Teff,logg,FeH,Av]))
					modphot = (Mbol - BC_i) + 5.0*np.log10(dist) - 5.0
					# lnlike.append(-0.5 * ((modphot-inphot[kk][0])**2.0)/(inphot[kk][1]**2.0))
					lnlike.append(((modphot-inphot[kk][0])/inphot[kk][1])**2.0)
		return lnlike

