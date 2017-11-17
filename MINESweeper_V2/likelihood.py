import numpy as np

class likelihood(object):
	"""docstring for likelihood"""
	def __init__(self,datadict,MISTinfo,**kwargs):
		super(likelihood, self).__init__()

		self.verbose = kwargs.get('verbose',True)

		ageweight = kwargs.get('ageweight',False)

		# the dictionary with all data for likelihood
		self.datadict = datadict

		# init the MIST models
		self.MISTgen = self._initMIST(model=MISTinfo['model'],stripeindex=MISTinfo['stripe'],ageweight=ageweight)

		# if there is photometry, init the NN for those bands
		if 'filterarray' in MISTinfo.keys():
			self.ANNfn = self._initphotnn(MISTinfo['filterarray'],nnpath=MISTinfo['nnpath'])
		else:
			self.ANNfn = None

	def _initMIST(self,model=None,stripeindex=None,ageweight=False):
		from .MISTmod import MISTgen
		# init MISTgen
		return MISTgen(model=model,stripeindex=stripeindex,ageweight=ageweight,verbose=self.verbose)

	def _initphotnn(self,filterarray,nnpath=None):
		from .photANN import ANN

		ANNdict = {}
		for ff in filterarray:
			try:
				ANNdict[ff] = ANN(ff,nnpath=nnpath,verbose=self.verbose)
			except IOError:
				print('Cannot find NN HDF5 file for {0}'.format(ff))
		return ANNdict
		
	def like(self,pars):
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
		self.MIST_i = self.MISTgen.getMIST(pars,verbose=False)

		if self.MIST_i == "ValueError":
			return -np.inf

		self.MIST_i['Teff'] = 10.0**self.MIST_i['log(Teff)']
		self.MIST_i['Rad']  = 10.0**self.MIST_i['log(Rad)']

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

			# fit a parallax if given
			if type(parallax) != type(None):
				parallax_i = 1000.0/dist
				lnlike += -0.5 * ((parallax[0]-parallax_i)**2.0)/(parallax[1]**2.0)

			if type(inphot) != type(None):

				# define some parameters from modMIST for the photomteric predictions
				Teff = 10.0**modMIST['log(Teff)']
				FeH  = modMIST['[Fe/H]']
				logg = modMIST['log(g)']

				# calc bolometric magnitude
				Mbol = -2.5*modMIST['log(L)']+4.74

				# for each filter, calculate predicted mag and calc likelihood
				for kk in inphot.keys():
					BC_i = float(self.ANNfn[kk].eval([Teff,logg,FeH,Av]))
					modphot = (Mbol - BC_i) + 5.0*np.log10(dist) - 5.0
					lnlike += -0.5 * ((modphot-inphot[kk][0])**2.0)/(inphot[kk][1]**2.0)

		return lnlike