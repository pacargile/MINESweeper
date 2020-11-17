import numpy as np
from .genmod import GenMod
from .fastMISTmod import GenMIST
from datetime import datetime

class likelihood(object):
     """docstring for likelihood"""
     def __init__(self,fitargs,fitpars,runbools,**kwargs):
          super(likelihood, self).__init__()

          self.verbose = kwargs.get('verbose',True)
          self.fitargs = fitargs

          # split up the boolean flags
          self.spec_bool = runbools[0]
          self.phot_bool = runbools[1]
          self.normspec_bool = runbools[2]
          self.photscale_bool = runbools[3]

          # run with weights such that d(EEP)/d(age) = constant
          self.ageweight = self.fitargs['ageweight']
          # array of parameters that are fixed
          self.fixedpars = self.fitargs['fixedpars']
          # turn on or off surface diffusion
          self.dif_bool  = self.fitargs['mistdif']

          # initialize the model generation class
          self.GM = GenMod()

          # initialize the ANN for spec and phot if user defined
          if self.spec_bool:
               self.GM._initspecnn(nnpath=fitargs['specANNpath'],
                    NNtype=self.fitargs['NNtype'])
          if self.phot_bool:
               self.GM._initphotnn(self.fitargs['obs_phot'].keys(),
                    nnpath=fitargs['photANNpath'])

          if fitpars[1]['EEP']:
               self.GMIST = GenMIST(MISTpath=self.fitargs['MISTpath'],
                    ageweight=self.ageweight)
               # self.GMIST._initMIST(ageweight=True,fast=False)

          # determine the number of dims
          self.ndim = 0
          self.fitpars_i = []
          for pp in fitpars[0]:
               if fitpars[1][pp]:
                    self.fitpars_i.append(pp)
                    self.ndim += 1


          # dictionary to translate par names to MIST names
          self.MISTrename = {
               'log_age':'log(Age)',
               'star_mass':'Mass',
               'log_R':'log(R)',
               'log_L':'log(L)',
               'log_Teff':'log(Teff)',
               'log_g':'log(g)',}

     def lnlikefn(self,pars):

          # build the parameter dictionary
          self.parsdict = {pp:vv for pp,vv in zip(self.fitpars_i,pars)} 

          # add fixed parameters to parsdict
          for kk in self.fixedpars.keys():
               self.parsdict[kk] = self.fixedpars[kk]
          
          # first check to see if EEP is in pars, if so query 
          # isochrones for stellar parameters
          if 'EEP' in self.parsdict.keys():
               MISTpred = self.GMIST.getMIST(
                    eep=self.parsdict['EEP'],
                    mass=self.parsdict['initial_Mass'],
                    feh=self.parsdict['initial_[Fe/H]'],
                    afe=self.parsdict['initial_[a/Fe]'],
                    verbose=False,
                    )
               if type(MISTpred) == type(None):
                    self.parsdict['Teff']   = np.inf
                    self.parsdict['log(g)'] = np.inf
                    self.parsdict['[Fe/H]'] = np.inf
                    self.parsdict['[a/Fe]'] = np.inf
                    self.parsdict['log(R)'] = np.inf

                    # add other paramters just for fun
                    self.parsdict['log(Age)'] = np.inf
                    self.parsdict['Agewgt']   = np.inf
                    self.parsdict['Mass']     = np.inf
                    self.parsdict['log(L)']   = np.inf
                    return -np.inf

               # stick MIST model pars into useful dict format
               MISTdict = ({
                    kk:pp for kk,pp in zip(
                         self.GMIST.modpararr,MISTpred)
                    })
               for kk in MISTdict.keys():
                    if kk in self.MISTrename.keys():
                         MISTdict[self.MISTrename[kk]] = MISTdict.pop(kk)

               self.parsdict['Teff']   = 10.0**MISTdict['log(Teff)']
               self.parsdict['log(g)'] = MISTdict['log(g)']
               self.parsdict['log(R)'] = MISTdict['log(R)']


               # use MIST predictions for surface abundances if asked for
               if self.dif_bool:
                    self.parsdict['[Fe/H]'] = MISTdict['[Fe/H]']
                    self.parsdict['[a/Fe]'] = MISTdict['[a/Fe]']
               else:
                    self.parsdict['[Fe/H]'] = MISTdict['initial_[Fe/H]']
                    self.parsdict['[a/Fe]'] = MISTdict['initial_[a/Fe]']

               # add other paramters just for fun
               self.parsdict['EEP'] = MISTdict['EEP']
               self.parsdict['log(Age)'] = MISTdict['log(Age)']
               self.parsdict['Mass'] = MISTdict['Mass']
               self.parsdict['log(L)'] = MISTdict['log(L)']

               if self.ageweight:
                    self.parsdict['Agewgt'] = MISTdict['Agewgt']

          if self.spec_bool:
               specpars = ([
                    self.parsdict[pp] 
                    if (pp in self.parsdict.keys()) else np.nan
                    for pp in ['Teff','log(g)','[Fe/H]','[a/Fe]','Vrad','Vrot','Vmic','Inst_R'] 
                    ])
               if self.normspec_bool:
                    specpars = specpars + [self.parsdict[pp] for pp in self.fitpars_i if 'pc' in pp]
          else:
               specpars = None

          if self.phot_bool:
               photpars = [self.parsdict[pp] for pp in ['Teff','log(g)','[Fe/H]','[a/Fe]']]
               if 'log(A)' in self.fitpars_i:
                    photpars = photpars + [self.parsdict['log(A)']]
               else:
                    photpars = photpars + [self.parsdict['log(R)'],self.parsdict['Dist']]
               photpars = photpars + [self.parsdict['Av']]
          else:
               photpars = None

          # calculate likelihood probability
          lnlike_i = self.lnlike(specpars=specpars,photpars=photpars)

          # add the prior on Agewgt
          if self.ageweight:
               lnlike_i += np.log(self.parsdict['Agewgt'])

          if lnlike_i == np.nan:
               print(pars,lnlike_i)

          return lnlike_i

     def lnlike(self,specpars=None,photpars=None):

          if self.spec_bool:
               # generate model spectrum
               specmod = self.GM.genspec(specpars,outwave=self.fitargs['obs_wave_fit'],normspec_bool=self.normspec_bool)
               modwave_i,modflux_i = specmod

               print(modwave_i)
               print(modflux_i)

               # calc chi-square for spec
               specchi2 = np.sum( 
                    [((m-o)**2.0)/(s**2.0) for m,o,s in zip(
                         modflux_i,self.fitargs['obs_flux_fit'],self.fitargs['obs_eflux_fit'])])
          else:
               specchi2 = 0.0

          if self.phot_bool:
               # generate model SED
               if self.photscale_bool:
                    sedmod  = self.GM.genphot_scaled(photpars)
               else:
                    sedmod  = self.GM.genphot(photpars)

               # calculate chi-square for SED
               sedchi2 = np.sum(
                    [((sedmod[kk]-self.fitargs['obs_phot'][kk][0])**2.0)/(self.fitargs['obs_phot'][kk][1]**2.0) 
                    for kk in self.fitargs['obs_phot'].keys()]
                    )
          else:
               sedchi2 = 0.0

          # print('LnL:',specpars,photpars,-0.5*(specchi2+sedchi2))

          # return ln(like) = -0.5 * chi-square
          return -0.5*(specchi2+sedchi2)
