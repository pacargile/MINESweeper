#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The top-level code to initialize and run the MINESweeper fitter
"""

import sys,time,datetime, math,re
from datetime import datetime
import numpy as np

try:
	import dynesty
except ImportError:
	print('Dynesty Not Installed, Will Throw an Error if Fitting Data!')
	pass

from .utils import DM
from .likelihood import likelihood
from .priors import priors

def lnprob(pars,likefn,priorfn):
	# first pass pars into priorfn
	lnprior = priorfn.likeprior(pars)

	# check to see if outside of a flat prior
	if lnprior == -np.inf:
		return -np.inf

	lnlike = likefn.like(pars)
	if lnlike == -np.inf:
		return -np.inf
	
	return lnprior + lnlike	


class MINESweeper(object):
	"""
	Class for MINESweeper
	"""
	def __init__(self):
		self.DM = DM
		self.priors = priors
		self.likelihood = likelihood

		self.outfilepars = (
			['EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]','log(Age)','Mass','Rad',
			'log(L)','Teff','[Fe/H]','log(g)']
			)

	def __call__(self,indicts):
		'''
		call instance so that run_dynesty can be called with multiprocessing
		and still have all of the class instance variables

		:params datadict:
			data dictionary containing all of the user defined data.
			Also contains parameters specific for dynesty sampling,
			e.g., number of active points

		'''
		self.run_dynesty(indicts)


	def run(self,*args,**kwargs):
		# check to make sure there is a datadict, can't fit otherwise
		if 'datadict' in kwargs:
			datadict = kwargs['datadict']
		else:
			print('NO USER DEFINED DATA DICT, NOTHING TO FIT!')
			raise IOError

		# define prior dict if user defined
		priordict = kwargs.get('priordict',None)

		# set the weighting scheme if user set
		self.ageweight = kwargs.get('ageweight',False)
		if self.ageweight:
			self.outfilepars.append('Agewgt')

		# init output file
		self.output = kwargs.get('output','Test.dat')

		# if isochrone == True: initialize MIST (this is the default for now)
		isochroneON = datadict.get('isochroneON',True)
		self.MISTinfo = {}
		if isochroneON:
			MISTmodel = datadict.get('MISTmodel',None)
			stripeindex = datadict.get('stripeindex',None)
			self.MISTinfo['model'] = MISTmodel
			self.MISTinfo['stripe'] = stripeindex

			self.ndim = 3

		# check to make sure user has included some sort of input data
		if ('pars' not in datadict.keys()) and ('phot' not in datadict.keys()) and ('spec' not in datadict.keys()):
			print('MUST HAVE SOME SORT OF INPUT DATA: pars, phot, or spec')
			raise IOError

		# check to see if user is using pars (e.g., datadict['pars']['Teff'] = [5770.0,150.0])
		inpars = datadict.get('pars',None)
		if type(inpars) != type(None):
			self.MISTinfo['pars'] = inpars.keys()

		# check to see if user is using SED data (e.g., datadict['phot']['2MASS_Ks'] = [13.0,0.01])
		inphot = datadict.get('phot',None)
		if type(inphot) != type(None):
			self.MISTinfo['filterarray'] = inphot.keys()
			self.outfilepars = self.outfilepars+['Dist','Av']
			self.ndim = 5

		# check to see if user is using a spectrum (e.g., datadict['spec']['WAVE','FLUX','err_FLUX'] = [...])
		if 'spec' in datadict:
			print('NOT SET UP YET TO DO SPEC FITTING')
			raise IOError
		
		# run the fitter
		self([datadict,priordict,self.MISTinfo])


	def run_dynesty(self,indict):

		# split input dict
		datadict = indict[0]
		priordict = indict[1]
		MISTinfo = indict[2]

		if 'sampler' in datadict.keys():
			samplerdict = datadict['sampler']
		else:
			samplerdict = {}

		# initialize output file
		self._initoutput()

		# initialize the prior class
		self.priorfn = self.priors(priordict)

		# initialize the likelihood class
		self.likefn = self.likelihood(datadict,MISTinfo,ageweight=self.ageweight)

		# pick random point within grid as starting active points

		# run sampler
		sampler = self.runsampler(samplerdict)

	def _initoutput(self):
		# init output file
		self.outff = open(self.output,'w')
		self.outff.write('Iter ')
		for kk in self.outfilepars:
			self.outff.write('{0} '.format(kk))
		self.outff.write('log(lk) log(vol) log(wt) h nc log(z) delta(log(z))')
		self.outff.write('\n')

	def runsampler(self,samplerdict):
		# pull out user defined sampler variables
		npoints = samplerdict.get('npoints',200)
		samplertype = samplerdict.get('samplertype','multi')
		bootstrap = samplerdict.get('bootstrap',0)
		update_interval = samplerdict.get('update_interval',float(self.ndim))
		samplemethod = samplerdict.get('samplemethod','unif')
		delta_logz_final = samplerdict.get('delta_logz_final',0.01)
		flushnum = samplerdict.get('flushnum',10)

		print(
			'Start Dynesty w/ {0} number of samples w/ stopping criteria of dlog(z) = {1}'.format(
				npoints,delta_logz_final))
		startmct = datetime.now()
		sys.stdout.flush()

		dy_sampler = dynesty.NestedSampler(
			lnprob,
			self.priorfn.prior_trans,
			self.ndim,
			logl_args=[self.likefn,self.priorfn],
			nlive=npoints,
			bound=samplertype,
			sample=samplemethod,
			update_interval=update_interval,
			bootstrap=bootstrap,
			)


		ncall = 0
		nit = 0

		for it, results in enumerate(dy_sampler.sample(dlogz=delta_logz_final)):
			(worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
				h, nc, worst_it, propidx, propiter, eff, delta_logz) = results			

			self.outff.write('{0} '.format(it))
			# self.outff.write(' '.join([str(q) for q in vstar]))
			# write parameters at iteration if not ValueError
			if self.likefn.MIST_i != 'ValueError':
				for pp in self.outfilepars:
					self.outff.write('{0} '.format(self.likefn.MIST_i[pp]))
			else:
				for VV in vstar:
					self.outff.write('{0} '.format(VV))
				for _ in range(len(self.outfilepars)-len(vstar)):
					self.outff.write('-999.99 ')

			self.outff.write(' {0} {1} {2} {3} {4} {5} {6} '.format(
				loglstar,logvol,logwt,h,nc,logz,delta_logz))
			self.outff.write('\n')

			ncall += nc
			if (it%flushnum) == 0:
				self.outff.flush()

				# format/output results
				if logz < -1e6:
					logz = -np.inf
				if delta_logz > 1e6:
					delta_logz = np.inf
				if logzvar >= 0.:
					logzerr = np.sqrt(logzvar)
				else:
					logzerr = np.nan
				if logzerr > 1e6:
					logzerr = np.inf
					
				sys.stdout.write("\riter: {0:d} | nc: {1:d} | ncall: {2:d} | eff(%): {3:6.3f} | "
					"logz: {4:6.3f} +/- {5:6.3f} | dlogz: {6:6.3f} > {7:6.3f}      "
					.format(nit + it, nc, ncall, eff, 
						logz, logzerr, delta_logz, delta_logz_final))
				sys.stdout.flush()

		# add live points to sampler object
		for it2, results in enumerate(dy_sampler.add_live_points()):
			# split up results
			(worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar,
			h, nc, worst_it, boundidx, bounditer, eff, delta_logz) = results

			self.outff.write('{0} '.format(it2))

			lnlike_i = self.likefn.like(vstar)

			# write parameters at iteration if not ValueError
			if self.likefn.MIST_i != 'ValueError':
				for pp in self.outfilepars:
					self.outff.write('{0} '.format(self.likefn.MIST_i[pp]))
			else:
				for VV in vstar:
					self.outff.write('{0} '.format(VV))
				for _ in range(len(self.outfilepars)-len(vstar)):
					self.outff.write('-999.99 ')

			self.outff.write(' {0} {1} {2} {3} {4} {5} {6} '.format(
				loglstar,logvol,logwt,h,nc,logz,delta_logz))
			self.outff.write('\n')

			ncall += nc

			# format/output results
			if logz < -1e6:
				logz = -np.inf
			if delta_logz > 1e6:
				delta_logz = np.inf
			if logzvar >= 0.:
				logzerr = np.sqrt(logzvar)
			else:
				logzerr = np.nan
			if logzerr > 1e6:
				logzerr = np.inf
			sys.stdout.write("\riter: {:d} | nc: {:d} | ncall: {:d} | eff(%): {:6.3f} | "
				"logz: {:6.3f} +/- {:6.3f} | dlogz: {:6.3f} > {:6.3f}      "
				.format(nit + it2, nc, ncall, eff, 
					logz, logzerr, delta_logz, delta_logz_final))

			sys.stdout.flush()

		self.outff.close()
		sys.stdout.write('\n')

		finishtime = datetime.now()
		print('RUN TIME: {0}'.format(finishtime-startmct))

		return dy_sampler		

	"""
	# write out final live points to output file
		self.res = dy_sampler.results
		for jj,vv in enumerate(self.res['samples'][-self.res['nlive']:]):
			niter = range(1,self.res['niter']+1)[jj]+it
			logl = self.res['logl'][-self.res['nlive']:][jj]
			logvol = self.res['logvol'][-self.res['nlive']:][jj]
			logwt = self.res['logwt'][-self.res['nlive']:][jj]
			h = self.res['information'][-self.res['nlive']:][jj]
			nc = self.res['ncall'][-self.res['nlive']:][jj]
			logz = self.res['logz'][-self.res['nlive']:][jj]
			delta_logz = delta_logz_final

			self.outff.write('{0} '.format(niter))
			# self.outff.write(' '.join([str(q) for q in vv]))
			lnlike_i = self.likefn.like(vstar)

			for pp in self.outfilepars:
				self.outff.write('{0} '.format(self.likefn.MIST_i[pp]))			
			self.outff.write(' {0} {1} {2} {3} {4} {5} {6} '.format(
				logl,logvol,logwt,h,nc,logz,delta_logz))
			self.outff.write('\n')
		"""


