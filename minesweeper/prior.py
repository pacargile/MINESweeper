import numpy as np
from scipy.stats import norm, truncnorm, expon, reciprocal,beta
from .advancedpriors import AdvancedPriors

class prior(object):
     """docstring for priors"""
     def __init__(self, fitargs, inpriordict,fitpars,runbools):
          super(prior, self).__init__()

          self.fitargs = fitargs
          self.fixedpars = self.fitargs['fixedpars']

          # determine the number of dims
          self.ndim = 0
          self.fitpars_i = []
          for pp in fitpars[0]:
               if fitpars[1][pp]:
                    self.fitpars_i.append(pp)
                    self.ndim += 1

          # set default advance prior bools to False
          self.imf_bool  = False
          self.gal_bool  = False
          self.galage_bool = False
          self.vrot_bool = False
          self.vtot_bool = False
          self.alpha_bool = False

          # see if there is an angular diameter
          if 'AngDia' in inpriordict.keys():
               self.angdia_bool = True
          else:
               self.angdia_bool = False


          # find uniform priors and put them into a 
          # dictionary used for the prior transformation
          self.priordict = {}
          self.priordict['uniform'] = {}
          self.priordict['gaussian'] = {}
          self.priordict['tgaussian'] = {}
          self.priordict['exp'] = {}
          self.priordict['texp'] = {}
          self.priordict['loguniform'] = {}
          self.priordict['beta'] = {}

          # put any additional priors into a dictionary so that
          # they can be applied in the lnprior_* functions
          self.additionalpriors = {}

          for kk in inpriordict.keys():
               if kk == 'blaze_coeff':
                    self.polycoefarr = inpriordict['blaze_coeff']
               elif kk == 'IMF':
                    self.imf_bool = True
                    self.imf = inpriordict['IMF']['IMF_type']
               elif kk == 'GAL':
                    self.gal_bool = True
                    self.lb_coords = inpriordict['GAL']['lb_coords']
                    if 'Dist' in inpriordict.keys():
                         self.mindist = inpriordict['Dist']['pv_uniform'][0]
                         self.maxdist = inpriordict['Dist']['pv_uniform'][1]
                    else:
                         self.mindist = 1.0
                         self.maxdist = 200000.0
               elif kk == 'GALAGE':
                    self.gal_bool = True
                    self.galage_bool = True
                    self.lb_coords = inpriordict['GALAGE']['lb_coords']
                    self.agepars = inpriordict['GALAGE']['pars']
                    if 'Dist' in inpriordict.keys():
                         self.mindist = inpriordict['Dist']['pv_uniform'][0]
                         self.maxdist = inpriordict['Dist']['pv_uniform'][1]
                    else:
                         self.mindist = 1.0
                         self.maxdist = 200000.0
               elif kk == 'VROT':
                    self.vrot_bool = True
                    self.vrotpars = inpriordict['VROT']
               elif kk == 'ALPHA':
                    self.alpha_bool = True
                    self.minalpha = inpriordict['ALPHA']['min']
               elif kk == 'VTOT':
                    self.pm_bool = True
                    self.pmra  = inpriordict['VTOT']['pmra']
                    self.pmdec = inpriordict['VTOT']['pmdec']                    
               else:
                    for ii in inpriordict[kk].keys():
                         if ii == 'pv_uniform':
                              self.priordict['uniform'][kk] = inpriordict[kk]['pv_uniform']
                         elif ii == 'pv_gaussian':
                              self.priordict['gaussian'][kk] = inpriordict[kk]['pv_gaussian']
                         elif ii == 'pv_tgaussian':
                              self.priordict['tgaussian'][kk] = inpriordict[kk]['pv_tgaussian']
                         elif ii == 'pv_exp':
                              self.priordict['exp'][kk] = inpriordict[kk]['pv_exp']
                         elif ii == 'pv_texp':
                              self.priordict['texp'][kk] = inpriordict[kk]['pv_texp']
                         elif ii == 'pv_loguniform':
                              self.priordict['loguniform'][kk] = inpriordict[kk]['pv_loguniform']
                         elif ii == 'pv_beta':
                              self.priordict['beta'][kk] = inpriordict[kk]['pv_beta']
                         else:
                              try:
                                   self.additionalpriors[kk][ii] = inpriordict[kk][ii]
                              except KeyError:
                                   self.additionalpriors[kk] = {ii:inpriordict[kk][ii]}

          # split up the boolean flags
          self.spec_bool = runbools[0]
          self.phot_bool = runbools[1]
          self.modpoly_bool = runbools[2]
          self.photscale_bool = runbools[3]

          # dictionary of default parameter ranges
          self.defaultpars = {}
          self.defaultpars['Teff']   = [3000.0,17000.0]
          self.defaultpars['log(g)'] = [-1.0,5.5]
          self.defaultpars['[Fe/H]'] = [-4.0,0.5]
          self.defaultpars['[a/Fe]'] = [-0.2,0.6]
          self.defaultpars['Vrad']   = [-700.0,700.0]
          self.defaultpars['Vrot']   = [0,30.0]
          self.defaultpars['Vmic']   = [0.5,3.0]
          self.defaultpars['Inst_R'] = [10000.0,60000.0]
          self.defaultpars['log(A)'] = [-3.0,7.0]
          self.defaultpars['log(R)'] = [-2.0,3.0]
          self.defaultpars['Dist']   = [1.0,200000.0]
          self.defaultpars['Av']     = [0.0,5.0]
          self.defaultpars['Rv']     = [2.0,5.0]
          self.defaultpars['EEP']    = [200,808]
          self.defaultpars['initial_Mass'] = [0.25,30.0]
          self.defaultpars['initial_[Fe/H]'] = [-4.0,0.5]
          self.defaultpars['initial_[a/Fe]'] = [-0.2,0.6]
          self.defaultpars['log(Age)'] = [9.0,10.2]
          self.defaultpars['Age'] = [1.0,14.0]

          APdict = {}
          if self.gal_bool:
               APdict['l'] = self.lb_coords[0]
               APdict['b'] = self.lb_coords[1]
               APdict['mindist'] = self.mindist/1000.0
               APdict['maxdist'] = self.maxdist/1000.0

          if self.angdia_bool:
               APdict['AngDia'] = inpriordict['AngDia']['gaussian']

          self.AP = AdvancedPriors(**APdict)


     def priortrans(self,upars):

          # build the parameter dictionary
          uparsdict = {pp:vv for pp,vv in zip(self.fitpars_i,upars)} 

          # first pass parameters to PT for MIST if EEP in uparsdict
          if 'EEP' in uparsdict.keys():
               mistPT = self.priortrans_mist(uparsdict)
          else:
               mistPT = {}

          if self.spec_bool:
               specPT = self.priortrans_spec(uparsdict)
          else:
               specPT = {}

          if self.phot_bool:
               photPT = self.priortrans_phot(uparsdict)
          else:
               photPT = {}

          outputPT = {**mistPT,**specPT,**photPT}

          # print('lnP:',outputPT)
          return [outputPT[pp] for pp in self.fitpars_i]

     def priortrans_mist(self,upars):
          outdict = {}

          for namepar in ['EEP','initial_Mass','initial_[Fe/H]','initial_[a/Fe]']:
               if namepar in upars.keys():
                    upars_i = upars[namepar]
                    if namepar in self.priordict['uniform'].keys():
                         par_i = (
                              (max(self.priordict['uniform'][namepar])-min(self.priordict['uniform'][namepar]))*upars_i + 
                              min(self.priordict['uniform'][namepar])
                              )
                    elif namepar in self.priordict['gaussian'].keys():
                         par_i = norm.ppf(upars_i,loc=self.priordict['gaussian'][namepar][0],scale=self.priordict['gaussian'][namepar][1])

                    elif namepar in self.priordict['tgaussian'].keys():
                         loc = self.priordict['tgaussian'][namepar][2]
                         scale = self.priordict['tgaussian'][namepar][3]
                         a = (self.priordict['tgaussian'][namepar][0] - loc) / scale
                         b = (self.priordict['tgaussian'][namepar][1] - loc) / scale
                         par_i = truncnorm.ppf(upars_i,a,b,loc=loc,scale=scale)
                         if par_i == np.inf:
                              par_i = self.priordict['tgaussian'][namepar][1]
                    elif namepar in self.priordict['beta'].keys():
                         a = self.priordict['beta'][namepar][0]
                         b = self.priordict['beta'][namepar][1]
                         loc = self.priordict['beta'][namepar][2]
                         scale = self.priordict['beta'][namepar][3]
                         par_i = beta.ppf(upars_i,a,b,loc=loc,scale=scale)
                    else:
                         par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

                    outdict[namepar] = par_i

          return outdict      


     def priortrans_spec(self,upars):
     
          # calcuate transformation from prior volume to parameter for all modeled parameters

          outdict = {}

          for namepar in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Vmic','Inst_R']:
               if namepar in upars.keys():
                    upars_i = upars[namepar]
                    if namepar in self.priordict['uniform'].keys():
                         par_i = (
                              (max(self.priordict['uniform'][namepar])-min(self.priordict['uniform'][namepar]))*upars_i + 
                              min(self.priordict['uniform'][namepar])
                              )
                    elif namepar in self.priordict['gaussian'].keys():
                         par_i = norm.ppf(upars_i,loc=self.priordict['gaussian'][namepar][0],scale=self.priordict['gaussian'][namepar][1])

                    elif namepar in self.priordict['tgaussian'].keys():
                         loc = self.priordict['tgaussian'][namepar][2]
                         scale = self.priordict['tgaussian'][namepar][3]
                         a = (self.priordict['tgaussian'][namepar][0] - loc) / scale
                         b = (self.priordict['tgaussian'][namepar][1] - loc) / scale
                         par_i = truncnorm.ppf(upars_i,a,b,loc=loc,scale=scale)
                         if par_i == np.inf:
                              par_i = self.priordict['tgaussian'][namepar][1]
                    elif namepar in self.priordict['beta'].keys():
                         a = self.priordict['beta'][namepar][0]
                         b = self.priordict['beta'][namepar][1]
                         loc = self.priordict['beta'][namepar][2]
                         scale = self.priordict['beta'][namepar][3]
                         par_i = beta.ppf(upars_i,a,b,loc=loc,scale=scale)
                    elif namepar in self.priordict['exp'].keys():
                         par_i = expon.ppf(upars_i,loc=self.priordict['exp'][namepar][0],scale=self.priordict['exp'][namepar][1])
                    else:
                         par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

                    outdict[namepar] = par_i

          # if fitting a blaze function, do transformation for polycoef
          pcarr = [x_i for x_i in upars.keys() if 'pc' in x_i]
          if len(pcarr) > 0:
               for pc_i in pcarr:
                    if pc_i == 'pc_0':
                         uspec_scale = upars['pc_0']
                         outdict['pc_0'] = (2.0 - 0.5)*uspec_scale + 0.5
                    else:
                         pcind = int(pc_i.split('_')[-1])
                         
                         # pcmax = self.polycoefarr[pcind][0]+3.0*self.polycoefarr[pcind][1]
                         # pcmin = self.polycoefarr[pcind][0]-3.0*self.polycoefarr[pcind][1]
                         # outdict[pc_i] = (pcmax-pcmin)*upars[pc_i] + pcmin
                         
                         # outdict[pc_i] = norm.ppf(upars[pc_i],loc=self.polycoefarr[pcind][0],scale=self.polycoefarr[pcind][1])

                         loc = self.polycoefarr[pcind][0]
                         scale = self.polycoefarr[pcind][1]
                         minval = loc - 5.0 * scale 
                         maxval = loc + 5.0 * scale
                         a = (minval - loc) / scale
                         b = (maxval - loc) / scale
                         outdict[pc_i] = truncnorm.ppf(upars[pc_i],a,b,loc=loc,scale=scale)


          return outdict

     def priortrans_phot(self,upars):

          outdict = {}


          # if only fitting the SED, pull Teff/logg/FeH and do prior transformation
          if not self.spec_bool:
               for namepar in ['Teff','log(g)','[Fe/H]','[a/Fe]']:
                    if namepar in upars.keys():
                         upars_i = upars[namepar]
                         if namepar in self.priordict['uniform'].keys():
                              par_i = (
                                   (max(self.priordict['uniform'][namepar])-min(self.priordict['uniform'][namepar]))*upars_i + 
                                   min(self.priordict['uniform'][namepar])
                                   )
                         elif namepar in self.priordict['gaussian'].keys():
                              par_i = norm.ppf(upars_i,loc=self.priordict['gaussian'][namepar][0],scale=self.priordict['gaussian'][namepar][1])

                         elif namepar in self.priordict['tgaussian'].keys():
                              loc = self.priordict['tgaussian'][namepar][2]
                              scale = self.priordict['tgaussian'][namepar][3]
                              a = (self.priordict['tgaussian'][namepar][0] - loc) / scale
                              b = (self.priordict['tgaussian'][namepar][1] - loc) / scale
                              par_i = truncnorm.ppf(upars_i,a,b,loc=loc,scale=scale)
                              if par_i == np.inf:
                                   par_i = self.priordict['tgaussian'][namepar][1]
                         elif namepar in self.priordict['beta'].keys():
                              a = self.priordict['beta'][namepar][0]
                              b = self.priordict['beta'][namepar][1]
                              loc = self.priordict['beta'][namepar][2]
                              scale = self.priordict['beta'][namepar][3]
                              par_i = beta.ppf(upars_i,a,b,loc=loc,scale=scale)

                         elif namepar in self.priordict['exp'].keys():
                              par_i = expon.ppf(upars_i,loc=self.priordict['exp'][namepar][0],scale=self.priordict['exp'][namepar][1])
                         else:
                              par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

                         outdict[namepar] = par_i

          isopars = ['log(A)','log(R)','Av','Rv','Dist']
          if self.gal_bool:
               if 'Dist' in upars.keys():
                    outdict['Dist'] = 1000.0*self.AP.gal_ppf(upars['Dist'])
                    isopars.remove('Dist')

          for namepar in isopars:
               if namepar in upars.keys():
                    upars_i = upars[namepar]

                    if namepar in self.priordict['uniform'].keys():
                         par_i = (
                              (max(self.priordict['uniform'][namepar])-min(self.priordict['uniform'][namepar]))*upars_i + 
                              min(self.priordict['uniform'][namepar])
                              )
                    elif namepar in self.priordict['gaussian'].keys():
                         par_i = norm.ppf(upars_i,loc=self.priordict['gaussian'][namepar][0],scale=self.priordict['gaussian'][namepar][1])

                    elif namepar in self.priordict['exp'].keys():
                         par_i = expon.ppf(upars_i,loc=self.priordict['exp'][namepar][0],scale=self.priordict['exp'][namepar][1])

                    elif namepar in self.priordict['tgaussian'].keys():
                         loc = self.priordict['tgaussian'][namepar][2]
                         scale = self.priordict['tgaussian'][namepar][3]
                         a = (self.priordict['tgaussian'][namepar][0] - loc) / scale
                         b = (self.priordict['tgaussian'][namepar][1] - loc) / scale
                         par_i = truncnorm.ppf(upars_i,a,b,loc=loc,scale=scale)
                         if par_i == np.inf:
                              par_i = self.priordict['tgaussian'][namepar][1]

                    elif namepar in self.priordict['texp'].keys():
                         loc = self.priordict['texp'][namepar][2]
                         scale = self.priordict['texp'][namepar][3]
                         a = (self.priordict['texp'][namepar][0] - loc) / scale
                         b = (self.priordict['texp'][namepar][1] - loc) / scale
                         par_i = truncexpon.ppf(upars_i,a,b,loc=loc,scale=scale)
                         if par_i == np.inf:
                              par_i = self.priordict['texp'][namepar][1]

                    elif namepar in self.priordict['beta'].keys():
                         a = self.priordict['beta'][namepar][0]
                         b = self.priordict['beta'][namepar][1]
                         loc = self.priordict['beta'][namepar][2]
                         scale = self.priordict['beta'][namepar][3]
                         par_i = beta.ppf(upars_i,a,b,loc=loc,scale=scale)

                    elif namepar in self.priordict['loguniform'].keys():
                         par_i = reciprocal.ppf(upars_i, self.priordict['loguniform'][namepar][0], self.priordict['loguniform'][namepar][1])
                    else:
                         par_i = (self.defaultpars[namepar][1]-self.defaultpars[namepar][0])*upars_i + self.defaultpars[namepar][0]

                    outdict[namepar] = par_i

          return outdict

     def lnpriorfn(self,pars):
          # determine if user passed a dictionary or a list
          if isinstance(pars,list):
               # build the parameter dictionary
               parsdict = {pp:vv for pp,vv in zip(self.fitpars_i,pars)} 
          else:
               parsdict = pars

          # add fixed parameters to parsdict
          for kk in self.fixedpars.keys():
               parsdict[kk] = self.fixedpars[kk]

          # Advanced Priors
          advPrior = 0.0
          if self.imf_bool:
               if 'initial_Mass' not in parsdict.keys():
                    if np.isfinite(parsdict['log(g)']) and np.isfinite(parsdict['log(R)']):
                         Mass = 10.0**parsdict['log(g)'] + 2.0 * parsdict['log(R)']
               else:
                    Mass = parsdict['initial_Mass']     
               advPrior += float(self.AP.imf_lnprior(Mass))
               print('IMF',advPrior)


          if self.gal_bool or self.galage_bool:
               if np.isfinite(parsdict['log(Age)']) & np.isfinite(parsdict['Dist']):
                    lnp_dist,comp = self.AP.gal_lnprior(parsdict['Dist']/1000.0,return_components=True)
                    # Compute component membership probabilities.
                    logp_thin  = comp['number_density'][0]
                    logp_thick = comp['number_density'][1]
                    logp_halo  = comp['number_density'][2]

                    lnprior_thin = logp_thin - lnp_dist
                    lnprior_thick = logp_thick - lnp_dist
                    lnprior_halo = logp_halo - lnp_dist
               
               if self.galage_bool:
                    # do galactic age prior in addition
                    lnp_age = self.AP.age_lnprior(
                         10.0**(parsdict['log(Age)']-9.0),
                         lnp_thin=lnprior_thin,
                         lnp_thick=lnprior_thick,
                         lnp_halo=lnprior_halo,
                         thin=self.agepars['thin'],
                         thick=self.agepars['thick'],
                         halo=self.agepars['halo'],
                         )
               else:
                    # no gal age prior, just denisty prior
                    lnp_age = 0.0

               print('GAL',advPrior)
               advPrior += (lnp_dist+lnp_age)

          # if self.gal_bool:
          #      if np.isfinite(parsdict['Dist']):
          #           advPrior += self.AP.gal_lnprior(parsdict['Dist']/1000.0,self.lb_coords)

          if self.vrot_bool:
               if 'initial_Mass' not in parsdict.keys():
                    if np.isfinite(parsdict['log(g)']) and np.isfinite(parsdict['log(R)']):
                         parsdict['initial_Mass'] = 10.0**(parsdict['log(g)'] + 2.0 * parsdict['log(R)'])
               advPrior += float(self.AP.vrot_lnprior(
                    vrot=parsdict['Vrot'],
                    mass=parsdict['initial_Mass'],
                    eep=parsdict['EEP'],
                    logg=parsdict['log(g)'],
                    dwarf=self.vrotpars['dwarf'],
                    giant=self.vrotpars['giant'],
                    ))
               print('VROT',advPrior)


          if self.vtot_bool:
               if 'Vrad' not in parsdict.keys():
                    vrad_i = 0.0
               else:
                    vrad_i = parsdict['Vrad']

               if np.isfinite(self.pmra) and np.isfinite(self.pmdec):
                    mu = np.sqrt( (self.pmra**2.0) + (self.pmdec**2.0) ) / 1000.0
               else:
                    mu = 0.0

               if np.isfinite(parsdict['Dist']):
                    dist = parsdict['Dist']
               else:
                    dist = 1e+6

               advPrior += float(
                    self.AP.Vtot_lnprior(
                         vrad=vrad_i,
                         mu=mu,
                         dist=dist
                         )
                    )

          if self.alpha_bool:
               if 'initial_[a/Fe]' in parsdict.keys():
                    afe_i = parsdict['initial_[a/Fe]']
               else:
                    afe_i = parsdict['[a/Fe]']

               advPrior += float(self.AP.alpha_lnprior(logg=parsdict['log(g)'],
                    aFe=afe_i,eep=parsdict['EEP'],minalpha=self.minalpha))

          # check to see if any priors in additionalprior dictionary,
          # save time by quickly returning zero if there are none
          if len(self.additionalpriors.keys()) == 0:
               return advPrior

          # MIST priors
          if 'EEP' in parsdict.keys():
               mistPrior = self.lnprior_mist(parsdict)
          else:
               mistPrior = 0.0

          # Spectra Priors
          if self.spec_bool:
               specPrior = self.lnprior_spec(parsdict)
               print(parsdict,specPrior)
          else:
               specPrior = 0.0

          # Phot Priors
          if self.phot_bool:
               photPrior = self.lnprior_phot(parsdict)
          else:
               photPrior = 0.0


          return mistPrior + specPrior + photPrior + advPrior

     def lnprior_mist(self,parsdict,verbose=True):
          lnprior = 0.0
          if len(self.additionalpriors.keys()) > 0:
               for kk in self.additionalpriors.keys():
                    lnprior_i = 0.0
                    if kk in ['EEP','initial_Mass','initial_[Fe/H]','initial_[a/Fe]','log(Age)','Age']:
                         if kk == 'Age':
                              parsdict['Age'] = 10.0**(parsdict['log(Age)']-9.0)
                         # if prior is Gaussian
                         if 'uniform' in self.additionalpriors[kk].keys():
                              if ((parsdict[kk] < self.additionalpriors[kk]['uniform'][0]) or 
                                   (parsdict[kk] > self.additionalpriors[kk]['uniform'][1])):
                                   return -np.inf
                         if 'gaussian' in self.additionalpriors[kk].keys():
                              lnprior_i += -0.5 * (((parsdict[kk]-self.additionalpriors[kk]['gaussian'][0])**2.0)/
                                   (self.additionalpriors[kk]['gaussian'][1]**2.0))
                         if 'tgaussian' in self.additionalpriors[kk].keys():
                              if ((parsdict[kk] < self.additionalpriors[kk]['tgaussian'][0]) or 
                                   (parsdict[kk] > self.additionalpriors[kk]['tgaussian'][1])):
                                   return -np.inf                              
                              lnprior_i += -0.5 * (((parsdict[kk]-self.additionalpriors[kk]['tgaussian'][2])**2.0)/
                                   (self.additionalpriors[kk]['tgaussian'][3]**2.0))
                         if 'beta' in self.additionalpriors[kk].keys():
                              raise IOError('Beta Prior not implimented yet!!!')
                         if 'log-normal' in self.additionalpriors[kk].keys():
                              raise IOError('Log-Normal Prior not implimented yet!!!')
                         try:
                             parsdict.pop("Age")
                         except KeyError:
                              pass
                    # print(kk,parsdict[kk],self.additionalpriors[kk],lnprior_i)
                    # try:
                    #      if np.isnan(lnprior_i):
                    #           print(kk,parsdict[kk],self.additionalpriors[kk],lnprior_i)
                    # except KeyError:
                    #      pass

                    lnprior += lnprior_i

          return lnprior                


     def lnprior_spec(self,parsdict,verbose=True):
          lnprior = 0.0

          # check to see if any of the parameter are included in additionalpriors dict
          if len(self.additionalpriors.keys()) > 0:
               for kk in self.additionalpriors.keys():
                    lnprior_i = 0.0
                    print(kk,parsdict[kk],self.additionalpriors[kk],lnprior_i)
                    # check to see if additional prior is for a spectroscopic parameter
                    if kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Vmic','Inst_R']:
                         # if prior is Gaussian
                         if 'uniform' in self.additionalpriors[kk].keys():
                              if ((parsdict[kk] < self.additionalpriors[kk]['uniform'][0]) or 
                                   (parsdict[kk] > self.additionalpriors[kk]['uniform'][1])):
                                   return -np.inf
                         if 'gaussian' in self.additionalpriors[kk].keys():
                              lnprior += -0.5 * (((parsdict[kk]-self.additionalpriors[kk]['gaussian'][0])**2.0)/
                                   (self.additionalpriors[kk]['gaussian'][1]**2.0))
                         if 'tgaussian' in self.additionalpriors[kk].keys():
                              if ((parsdict[kk] < self.additionalpriors[kk]['tgaussian'][0]) or 
                                   (parsdict[kk] > self.additionalpriors[kk]['tgaussian'][1])):
                                   return -np.inf                              
                         if 'beta' in self.additionalpriors[kk].keys():
                              raise IOError('Beta Prior not implimented yet!!!')
                         if 'log-normal' in self.additionalpriors[kk].keys():
                              raise IOError('Log-Normal Prior not implimented yet!!!')

          # # if fitting a blaze function, then check for additional priors
          # if self.modpoly_bool:
          #    for pp in self.fitpars_i:
          #         if pp[:2] == 'pc_':
          #              lnprior += -0.5 * ((parsdict[pp]/self.polycoefarr[kk][1])**2.0)

                    lnprior += lnprior_i
          return lnprior

     def lnprior_phot(self,parsdict,verbose=True):
          lnprior = 0.0

          parsdict_i = {}
          # if only fitting the SED, pull Teff/logg/FeH and do prior 
          if not self.spec_bool:
               parsdict_i['Teff']   = parsdict['Teff']
               parsdict_i['log(g)'] = parsdict['log(g)']
               parsdict_i['[Fe/H]'] = parsdict['[Fe/H]']
               parsdict_i['[a/Fe]'] = parsdict['[a/Fe]']
          if 'log(R)' in self.fitpars_i:
               parsdict_i['log(R)'] = parsdict['log(R)']
          if 'Dist' in self.fitpars_i:
               parsdict_i['Dist'] = parsdict['Dist']
          if 'log(A)' in self.fitpars_i:
               parsdict_i['log(A)'] = parsdict['log(A)']
          if 'Av' in self.fitpars_i:
               parsdict_i['Av'] = parsdict['Av']
          if 'Dist' in self.fitpars_i:
               parsdict_i['Parallax'] = 1000.0/parsdict['Dist']

          # check to see if any of these parameter are included in additionalpriors dict
          if len(self.additionalpriors.keys()) > 0:
               for kk in self.additionalpriors.keys():
                    lnprior_i = 0.0
                    if kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','log(R)','Dist','log(A)','Av','Parallax']:
                         # if prior is Gaussian
                         if 'uniform' in self.additionalpriors[kk].keys():
                              if ((parsdict_i[kk] < self.additionalpriors[kk]['uniform'][0]) or 
                                   (parsdict_i[kk] > self.additionalpriors[kk]['uniform'][1])):
                                   return -np.inf
                         if 'gaussian' in self.additionalpriors[kk].keys():
                              lnprior_i += -0.5 * (((parsdict_i[kk]-self.additionalpriors[kk]['gaussian'][0])**2.0)/
                                   (self.additionalpriors[kk]['gaussian'][1]**2.0))
                         if 'tgaussian' in self.additionalpriors[kk].keys():
                              if ((parsdict[kk] < self.additionalpriors[kk]['tgaussian'][0]) or 
                                   (parsdict[kk] > self.additionalpriors[kk]['tgaussian'][1])):
                                   return -np.inf                              
                         if 'beta' in self.additionalpriors[kk].keys():
                              raise IOError('Beta Prior not implimented yet!!!')
                         if 'log-normal' in self.additionalpriors[kk].keys():
                              raise IOError('Log-Normal Prior not implimented yet!!!')
                    lnprior += lnprior_i
          return lnprior
