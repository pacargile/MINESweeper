import numpy as np

class priors(object):
	"""docstring for priors"""
	def __init__(self, priordict):
		super(priors, self).__init__()
		self.priordict = priordict

		self.flatpriors = {}
		for kk in self.priordict.keys():
			self.flatpriors[kk] = self.priordict[kk]['noninform']

		self.addpriors = {}
		for kk in self.priordict.keys():
			if any([False if (x == 'noninform') else True for x in self.priordict[kk].keys()]):
				self.addpriors[kk] = self.priordict[kk]


	def prior_trans(self,par):
		""" de-project values from the prior unit cube to parameter space """

		# EEP
		eep  = par[0]
		ueep = (self.flatpriors['EEP'][1]-self.flatpriors['EEP'][0])*eep + self.flatpriors['EEP'][0]

		# Age
		mass  = par[1]
		umass = (self.flatpriors['initial_mass'][1]-self.flatpriors['initial_mass'][0])*mass + self.flatpriors['initial_mass'][0]

		# FeH_in
		FeH  = par[2]
		uFeH = (self.flatpriors['initial_[Fe/H]'][1]-self.flatpriors['initial_[Fe/H]'][0])*FeH + self.flatpriors['initial_[Fe/H]'][0]

		uoutarr = [ueep,umass,uFeH]

		# check to see if Dist and Av are included in noninfom dict (i.e., fitting phot)
		if ('Dist' in self.flatpriors.keys()) & ('Av' in self.flatpriors.keys()):
			dist = par[3]
			Av = par[4]
			udist = (self.flatpriors['Dist'][1]-self.flatpriors['Dist'][0])*dist + self.flatpriors['Dist'][0]
			uAv   = (self.flatpriors['Av'][1]-self.flatpriors['Av'][0])*Av + self.flatpriors['Av'][0]

			uoutarr.append(udist)
			uoutarr.append(uAv)

		return uoutarr

	def prior_inversetrans(self,par):
		""" project the parameters onto the prior unit cube """	

		# EEP
		ueep  = par[0]
		eep = (ueep - self.flatpriors['EEP'][0])/(self.flatpriors['EEP'][1]-self.flatpriors['EEP'][0])

		# Age
		umass  = par[1]
		mass = (umass - self.flatpriors['initial_mass'][0])/(self.flatpriors['initial_mass'][1]-self.flatpriors['initial_mass'][0])

		# FeH_in
		uFeH  = par[2]
		FeH = (uFeH - self.flatpriors['initial_[Fe/H]'][0])/(self.flatpriors['initial_[Fe/H]'][1]-self.flatpriors['initial_[Fe/H]'][0])

		outarr = [eep,mass,FeH]

		# check to see if Dist and Av are included in noninfom dict (i.e., fitting phot)
		if ('Dist' in self.flatpriors.keys()) & ('Av' in self.flatpriors.keys()):
			udist = par[3]
			uAv = par[4]
			dist = (udist - self.flatpriors['Dist'][0])/(self.flatpriors['Dist'][1]-self.flatpriors['Dist'][0])
			Av   = (uAv - self.flatpriors['Av'][0])/(self.flatpriors['Av'][1]-self.flatpriors['Av'][0])

			outarr.append(dist)
			outarr.append(Av)

		return outarr

	def likeprior(self,par):
		""" additional priors to account for anything not included in prior_trans """
		lnprior = 0.0

		eep = par[0]
		mass = par[1]
		FeH = par[2]

		parnamearr = ['EEP','initial_mass','initial_[Fe/H]']
		pararr = [eep,mass,FeH]

		if len(par) > 3:
			dist = par[3]
			Av   = par[4]
			parnamearr = parnamearr + ['Dist','Av']
			pararr = pararr + [dist,Av]

		# check if EEP, Age, [Fe/H]in, Av, or Dist has an additional prior
		for name_i,par_i in zip(parnamearr,pararr):
			# check to see if par is in addpriors
			if name_i in self.addpriors.keys():
				# conditional on the type of prior
				if 'gaussian' in self.addpriors[name_i].keys():
					lnprior += -0.5 * (((par_i-self.addpriors[name_i]['gaussian'][0])**2.0)/
						(self.addpriors[name_i]['gaussian'][1]**2.0))
				elif 'broken' in self.addpriors[name_i].keys():
					if par_i < self.addpriors[name_i]['broken'][0]:
						return -np.inf
					elif (par > self.addpriors[name_i]['broken'][2]):
						# exponential decay with 10% e-fold 
						lnprior += -1.0 * (0.1)
					else:
						pass
				elif 'flat' in self.addpriors[name_i].keys():
					if (par_i < self.addpriors[name_i]['flat'][0]) or (par_i > self.addpriors[name_i]['flat'][1]):
						return -np.inf
				elif 'beta' in self.addpriors['EEP'].keys():
					raise IOError('Beta Prior not implimented yet!!!')
				elif 'log-normal' in self.addpriors['EEP'].keys():
					raise IOError('Log-Normal Prior not implimented yet!!!')
				else:
					pass

		return lnprior