import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

import glob, sys, pathlib,os
import itertools
import argparse
import json

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.stats import scoreatpercentile,reciprocal,truncnorm

from Payne.utils.quantiles import quantile

from minesweeper.fitutils import airtovacuum
from minesweeper.fitutils import SEDopt
from minesweeper.advancedpriors import AdvancedPriors

SPECNN = '/Users/pcargile/Astro/ThePayne/YSdata/YSTANN.h5'
PHOTNN = '/Users/pcargile/Astro/ThePayne/SED/VARRV/'
MISTISO = '/Users/pcargile/Astro/MIST/MIST_v2.0_spot/MIST_2.0_spot_EEPtrk_small.h5'

class postproc(object):
     """docstring for postproc"""
     def __init__(self, *args, **kwargs):
          super(postproc, self).__init__()

          self.AP = AdvancedPriors()

          from minesweeper import genmod
          self.GM = genmod.GenMod()
          self.GM._initspecnn(nnpath=SPECNN,NNtype='YST1')
          self.GM._initphotnn(filterarray=None,nnpath=PHOTNN)
          print('Using Spec ANN: {}'.format(SPECNN))
          print('Using Phot ANN: {}'.format(PHOTNN))


     def __call__(self,*args,**kwargs):

          starname = kwargs.get('starname','test')

          samplerpath = kwargs.get('samplerpath','MSoutput')
          pdfpath     = kwargs.get('pdfpath','MSpdf')

          inspec = Table.read('./spectra/{0}.fits'.format(starname),format='fits')
          inphot = Table.read('./phot/18Sco_phot.dat',format='ascii')
          indata = Table.read('./etc/other.dat',format='ascii')
          insamp = Table.read('./etc/samplerinfo.dat',format='ascii')

          # turn some data into more useful formats
          indata_i = {}
          for x in indata:
              indata_i[x['parameter']] = [x['value'],x['err']]

          insamp_i = {}
          for x in insamp:
              try:
                  par = json.loads(x['setting'])
              except:
                  par = x['setting']

              insamp_i[x['parameter']] = par

          print('Running... {}'.format(starname))

          samplefile = '{0}/{1}.dat'.format(samplerpath,starname)
          print('--> Looking for sample file: {}'.format(samplefile))

          try:
               SAMPLEStab = Table.read(samplefile,format='ascii')
          except:
               SAMPLEStab = Table.read(samplefile,format='ascii',data_end=-1)

          if len(SAMPLEStab) == 0:
               print('--> Problem with input file, len == 0')
               return

          # check to see if any old style column names
          replacecolname = ({
                    'logg':'log(g)',
                    'FeH':'[Fe/H]',               
                    'aFe':'[a/Fe]',
                    'init_FeH':'initial_[Fe/H]',
                    'init_Mass':'initial_Mass',
                    'logR':'log(R)',
                    'logAge':'log(Age)',
                    'logL':'log(L)',
                    'logA':'log(A)',
                    'Vrot':'Vstellar',
                    })

          for kk in replacecolname.keys():
               if kk in SAMPLEStab.keys():
                    SAMPLEStab[kk].name = replacecolname[kk]


          # pull phot data from input file
          self.photdata = {}
          for inphot_i in inphot:
              filt = inphot_i['filter']
              phot = inphot_i['phot']
              err  = inphot_i['err']

              if np.isfinite(phot):
                  # Add a noise floor for photometry
                  if filt.split('_')[0] == 'GaiaEDR3':
                      err = np.sqrt(err**2.0 + 0.05**2.0)
                  elif filt.split('_')[0] == '2MASS':
                      err = np.sqrt(err**2.0 + 0.02**2.0)
                  elif filt.split('_')[0] == 'WISE':
                      err = np.sqrt(err**2.0 + 0.02**2.0)
                  else:
                      pass

                  self.photdata[filt] = [phot,err]


          inputdict = {}
          inputdict['spec'] = {}
          inputdict['spec']['obs_wave']   = inspec['waveobs']
          inputdict['spec']['obs_flux']   = inspec['flux']
          inputdict['spec']['obs_eflux']  = inspec['err']


          self.medflux = np.nanmedian(inputdict['spec']['obs_flux'])
          # convert observed wavelengths to vacuum
          inputdict['spec']['obs_wave'] = airtovacuum(inputdict['spec']['obs_wave'])

          self.spec = inputdict['spec']

          pararr = SAMPLEStab.keys()
          pararr.remove('Iter')
          pararr.remove('log(lk)')
          pararr.remove('log(vol)')
          pararr.remove('log(wt)')
          pararr.remove('h')
          pararr.remove('nc')
          pararr.remove('log(z)')
          pararr.remove('delta(log(z))')

          # if sampling in Distance, make parallax col too and make dist into kpc
          if 'Dist' in SAMPLEStab.keys():
               SAMPLEStab['Para'] = 1000.0/SAMPLEStab['Dist']
               pararr.append('Para')

          if 'Inst_R' in SAMPLEStab.keys():
               SAMPLEStab['Inst_R'] = SAMPLEStab['Inst_R']/1000.0

          SAMPLEStab['Prob'] = np.exp(SAMPLEStab['log(wt)']-SAMPLEStab['log(z)'][-1])
          # SAMPLEStab['Prob'] = SAMPLEStab['Prob']/SAMPLEStab['Prob'].sum()
          # SAMPLEStab = SAMPLEStab[SAMPLEStab['Prob'] > 1E-6]
          
          outstat = self.genstat(
               SAMPLEStab,
               pararr,
               wgt=SAMPLEStab['Prob']
               )

          self.maxlike = np.argmax(SAMPLEStab['Prob'])
          self.bfdict = {par_i:outstat[par_i][1] for par_i in pararr}

          # sort pararr into spectro, phot, and iso pars
          self.pararr = ['Teff','log(g)','[Fe/H]','[a/Fe]']
          if 'Vrad' in SAMPLEStab.keys():
               self.pararr = self.pararr + ['Vrad','Vstellar','Inst_R']

          if 'Av' in SAMPLEStab.keys():
               if 'log(R)' in SAMPLEStab.keys():
                    self.pararr = self.pararr + ['log(R)','Dist','Av','Para']
               if 'log(A)' in SAMPLEStab.keys():
                    self.pararr = self.pararr + ['log(A)','Av']

          self.pararr = self.pararr + (
               ['EEP','log(Age)','initial_[Fe/H]','initial_[a/Fe]','initial_Mass',
               'Mass','log(L)'])

          self.axinfo = {kk:[kk,ii] for ii,kk in enumerate(self.pararr)}

          outarr = {}
          outarr['lnP'] = np.log(max(SAMPLEStab['Prob']))
          outarr['lnL'] = max(SAMPLEStab['log(lk)'])
          outarr['lnZ'] = float(SAMPLEStab['log(z)'][-1])

          # build parameter string to add to plot
          parstring = 'Star: {}\n'.format(starname)

          parstring_par = ({
               'Teff':   [r'T$_{eff}$ =',' {0:.1f} +{1:.1f}/-{2:.1f} ({3:.1f})\n'],
               'log(g)': [r'log(g) =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'Vrad':   [r'V$_{rad}$ =',' {0:.2f} +{1:.2f}/-{2:.2f} ({3:.2f})\n'],
               'Vstellar':   [r'V$_{stellar}$ =',' {0:.2f} +{1:.2f}/-{2:.2f} ({3:.2f})\n'],
               'Dist':   [r'Dist =',' {0:.1f} +{1:.1f}/-{2:.1f} ({3:.1f})\n'],
               'Av':     [r'A$_{v}$ =',' {0:.1f} +{1:.1f}/-{2:.1f} ({3:.1f})\n'],
               'log(Age)':       [r'log(Age) =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'initial_[Fe/H]': [r'[Fe/H]$_{i}$ =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'initial_[a/Fe]': [r'[a/Fe]$_{i}$ =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'initial_Mass':   [r'Mass$_{i}$ =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'log(L)':         [r'log(L) =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n']})

          for kk in self.pararr:
               outstat_i = outstat[kk]
               parmed = outstat_i[1]
               parul  = outstat_i[2]-parmed
               parll  = parmed-outstat_i[0]
               parstd = outstat_i[3]

               outarr[kk] = [self.bfdict[kk],parmed,parul,parll,parstd]

               if kk in parstring_par.keys():
                    if kk in indata_i.keys():
                         truth = indata_i[kk][0]
                    else:
                         truth = np.nan
                    parstring = (
                         parstring + parstring_par[kk][0] + parstring_par[kk][1].format(
                              self.bfdict[kk],parul,parll,truth))

               self.axinfo[kk].append(outstat_i[4])
               self.axinfo[kk].append([self.bfdict[kk],parll,parul])

          pltpars = ([
               'Teff','log(g)','Vrad','Vstellar','Dist','Para',
               'Av','EEP','log(Age)','initial_[Fe/H]','initial_[a/Fe]','initial_Mass','log(L)'
               ])

          fig = plt.figure(figsize=(15,15))

          self.mkcornerfig(fig,pltpars,SAMPLEStab,truth=indata_i)

          axspec = fig.add_axes([0.3,0.75,0.65,0.15])
          axspecr = fig.add_axes([0.3,0.90,0.65,0.05])

          self.mkspec(axspec,axspecr,SAMPLEStab)

          axSED = fig.add_axes([0.62,0.54,0.325,0.175])
          axMAG = fig.add_axes([0.62,0.43,0.325,0.10])

          # self.mksed(axSED,axMAG,SAMPLEStab)

          plt.figtext(0.785,0.25,parstring,fontsize=10)

          outputfile = '{0}/{1}.png'.format(pdfpath,starname)
          fig.savefig(outputfile,dpi=250)
          plt.close(fig)
          return outarr


     def genstat(self,intab,pararr,wgt=None):
          outdict = {}
          for kk in pararr:
               outdict[kk] = quantile(intab[kk],[0.16,0.5,0.84],weights=wgt)
               avgwgt = np.average(intab[kk],weights=wgt)
               outdict[kk].append(
                    np.sqrt(np.average((intab[kk]-avgwgt)**2.0,weights=wgt))
                    )
               pltrange = quantile(intab[kk],[0.001,0.999],weights=wgt)
               pltrange[0] = pltrange[0] - 0.15*(pltrange[1]-pltrange[0])
               pltrange[1] = pltrange[1] + 0.15*(pltrange[1]-pltrange[0])

               outdict[kk].append(pltrange)
          return outdict

     def mkcornerfig(self,fig,pltpars,samples,truth={}):
          pararr_i = np.array([x for x in self.pararr if x in pltpars])
          parind_i = np.array(range(len(pararr_i)))

          gs = gridspec.GridSpec(len(pararr_i),len(pararr_i))
          gs.update(wspace=0.05,hspace=0.05)


          for kk in itertools.product(pararr_i,pararr_i):
               kkind1 = parind_i[pararr_i == kk[0]][0]
               kkind2 = parind_i[pararr_i == kk[1]][0]
               axinfo_1 = self.axinfo[kk[0]]
               axinfo_2 = self.axinfo[kk[1]]
               xarr_range = axinfo_1[2]
               yarr_range = axinfo_2[2]
               ax = fig.add_subplot(gs[kkind1,kkind2])
               if kkind1 < kkind2:
                    ax.set_axis_off()
                    continue
               if kk[0] == kk[1]:

                    n,bins,_ = ax.hist(
                         samples[kk[0]],
                         bins=20,
                         histtype='step',
                         linewidth=2.0,
                         density=True,
                         range=xarr_range,
                         weights=samples['Prob'],
                         )
                    if kk[0] in truth.keys():
                         ax.axvline(
                              x=truth[kk[0]][0],
                              c='k',
                              lw=4.0,
                              alpha=0.75,
                              )
                    # # plot priors
                    # if kk[0] == 'Inst_R':
                    #      xarr = np.linspace(xarr_range[0],xarr_range[1],200)
                    #      yarr = np.exp( -0.5*((xarr-32.0)**2.0)/(0.15**2.0))
                    #      yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                    #      ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                    if kk[0] == 'Para':
                         xarr = np.linspace(xarr_range[0],xarr_range[1],200)
                         yarr = np.exp( -0.5*((xarr-truth['Parallax'][0])**2.0)/(truth['Parallax'][1]**2.0) )
                         yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                         ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                    if (kk[0] == 'initial_Mass'):
                         xarr = np.linspace(xarr_range[0],xarr_range[1],100)
                         yarr = self.AP.imf_lnprior(xarr)
                         yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                         ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                    # if (kk[0] == 'Dist'):
                    #      xarr = np.linspace(xarr_range[0],xarr_range[1],100)
                    #      yarr = np.exp( -0.5*((xarr-4000.0)**2.0)/(1000.0**2.0))
                    #      if all(yarr == 1):
                    #           yarr = yarr * n.max()
                    #      else:
                    #           yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                    #      ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                    # if (kk[0] == 'Av'):
                    #      xarr = np.linspace(xarr_range[0],xarr_range[1],100)
                    #      av_i = 0.04 * 3.1
                    #      yarr = np.exp( -0.5*((xarr-av_i)**2.0)/((av_i*0.15)**2.0))
                    #      if all(yarr == 1):
                    #           yarr = yarr * n.max()
                    #      else:
                    #           yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                    #      ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                    # if (kk[0] == 'Vrot'):
                    #      xarr = np.linspace(xarr_range[0],xarr_range[1],100)
                    #      yarr = np.exp(np.array([self.AP.vrot_lnprior(
                    #           vrot=x_i,
                    #           teff=self.bfdict['Teff'],
                    #           mass=self.bfdict['initial_Mass'],
                    #           lograd=self.bfdict['log(R)'],
                    #           eep=self.bfdict['EEP'],
                    #           ) for x_i in xarr]))

                    #      if all(yarr == 1):
                    #           yarr = yarr * n.max()
                    #      else:
                    #           yarr = n.max()*(yarr-yarr.min())/(yarr.max()-yarr.min())
                    #      ax.plot(xarr,yarr,ls='-',lw=1.0,c='green')

                    ax.axvline(x=self.bfdict[kk[0]],c='m')

                    ax.set_xlim(xarr_range)
                    ylimtmp = ax.get_ylim()
                    ax.set_ylim(ylimtmp[0],1.25*ylimtmp[1])

                    ax.set_yticks([])
                    if kk[0] != pararr_i[-1]:
                         ax.set_xticks([])
                    else:
                         ax.set_xlabel(kk[0])
               else:
                    ax.hist2d(
                         samples[kk[1]],
                         samples[kk[0]],
                         bins=20,
                         cmap='Blues',
                         range=[yarr_range,xarr_range],
                         weights=samples['Prob'],
                         )

                    if (kk[0] in truth.keys()) and (kk[1] in truth.keys()):
                         ax.scatter(
                              truth[kk[1]][0],truth[kk[0]][0],
                              marker='*',
                              c='k',
                              alpha=0.75,
                              )

                    ax.set_xlim(yarr_range)
                    ax.set_ylim(xarr_range)

               ax.xaxis.set_major_locator(MaxNLocator(4))
               ax.yaxis.set_major_locator(MaxNLocator(4))
               [l.set_rotation(45) for l in ax.get_xticklabels()]
               [l.set_fontsize(6) for l in ax.get_xticklabels()]
               [l.set_fontsize(6) for l in ax.get_yticklabels()]

               labelcol = 'k'

               if not ax.is_first_col():
                    ax.set_yticks([])
               elif ax.is_first_col() & ax.is_first_row():
                    ax.set_yticks([])
               elif kk[0] == pararr_i[0]:
                    pass
               else:
                    if 'initial' in kk[0]:
                         ax.set_ylabel(kk[0].split('_')[1]+r'$_{i}$')
                    elif kk[0] == '[a/Fe]':
                         ax.set_ylabel('['+r'$\alpha$'+'/Fe]')
                    elif kk[0] == 'Teff':
                         ax.set_ylabel(r'T$_{eff}$')
                    elif kk[0] == 'Dist':
                         ax.set_ylabel('Dist.')
                    elif kk[0] == 'Vrad':
                         ax.set_ylabel(r'V$_{rad}$')
                    elif kk[0] == 'Vstellar':
                         ax.set_ylabel(r'V$_{\bigstar}$')
                    else:
                         ax.set_ylabel(kk[0])

                    # if 'initial' in kk[0]:
                    #      ax.set_ylabel(kk[0].split('_')[1]+r'$_{i}$')
                    # elif kk[0] == '[a/Fe]':
                    #      ax.set_ylabel('['+r'$\alpha$'+'/Fe]')
                    # else:
                    #      ax.set_ylabel(kk[0])
                    if kk[0] in ['Teff','log(g)','initial_[Fe/H]','log(R)','log(Age)','initial_Mass','log(L)','Dist']:
                         labelcol = 'g'
                    ax.yaxis.label.set_color(labelcol)

               if not ax.is_last_row():
                    ax.set_xticks([])
               else:
                    if 'initial' in kk[1]:
                         ax.set_xlabel(kk[1].split('_')[1]+r'$_{i}$')
                    elif kk[1] == '[a/Fe]':
                         ax.set_xlabel('['+r'$\alpha$'+'/Fe]')
                    elif kk[1] == 'Teff':
                         ax.set_xlabel(r'T$_{eff}$')
                    elif kk[1] == 'Dist':
                         ax.set_xlabel('Dist.')
                    elif kk[1] == 'Vrad':
                         ax.set_xlabel(r'V$_{rad}$')
                    elif kk[1] == 'Vstellar':
                         ax.set_xlabel(r'V$_{\bigstar}$')
                    else:
                         ax.set_xlabel(kk[1])
                    if kk[1] in ['Teff','log(g)','initial_[Fe/H]','log(R)','log(Age)','initial_Mass','log(L)','Dist']:
                         labelcol = 'g'
                    ax.xaxis.label.set_color(labelcol)
          fig.align_labels()

     def mkspec(self,axspec,axspecr,samples):

          axspec.plot(
               self.spec['obs_wave'],
               self.spec['obs_flux'],
               ls='-',lw=1.25,c='k',zorder=0)

          pars_bf = ([
               self.bfdict['Teff'],
               self.bfdict['log(g)'],
               self.bfdict['[Fe/H]'],
               self.bfdict['[a/Fe]'],
               self.bfdict['Vrad'],
               self.bfdict['Vstellar'],
               np.nan,
               self.bfdict['Inst_R']*1000.0,
               ])

          pcarr = []
          for kk in samples.keys():
               if kk.split('_')[0] == 'pc':
                    pcarr.append(kk)
          pcarr.sort()
          for kk in pcarr:
               pars_bf.append(samples[kk][self.maxlike])

          modwave_bf,modflux_bf = self.GM.genspec(
               pars_bf,
               outwave=self.spec['obs_wave'],
               modpoly=False,
               )
          
          modflux_bf = modflux_bf#*self.medflux

          axspec.plot(
               modwave_bf,modflux_bf,
               ls='-',
               lw=2.0,
               c='m',
               alpha=0.75,
               zorder=1,
               )

          axspecr.plot(
               modwave_bf,(self.spec['obs_flux']-modflux_bf)/modflux_bf,
               ls='-',
               lw=1.0,
               c='m',
               alpha=0.75,
               zorder=1,
               )

          specmin = scoreatpercentile(self.spec['obs_flux'],0.5)
          specmax = scoreatpercentile(self.spec['obs_flux'],99.5)

          specrmin = scoreatpercentile((self.spec['obs_flux']-modflux_bf)/modflux_bf,0.5)
          specrmax = scoreatpercentile((self.spec['obs_flux']-modflux_bf)/modflux_bf,99.5)

          """
          for SAMPLES_i in np.random.choice(
               self.SAMPLEStab,50,
               p=self.SAMPLEStab['Prob']/self.SAMPLEStab['Prob'].sum()
               ):
               try:
                    pars = ([
                         SAMPLES_i['Teff'],
                         SAMPLES_i['log(g)'],
                         SAMPLES_i['[Fe/H]'],
                         SAMPLES_i['[a/Fe]'],
                         SAMPLES_i['Vrad'],
                         SAMPLES_i['Vrot'],
                         SAMPLES_i['Inst_R']*1000.0,
                         ])
               except (KeyError,ValueError):
                    pars = ([
                         SAMPLES_i['Teff'],
                         SAMPLES_i['log(g)'],
                         SAMPLES_i['[Fe/H]'],
                         SAMPLES_i['[a/Fe]'],
                         0.0,
                         0.0,
                         32000.0,
                         ])

               for kk in self.SAMPLEStab.keys():
                    if kk.split('_')[0] == 'pc':
                         pars.append(SAMPLES_i[kk])

               modwave,modflux = self.GM.genspec(
                    pars,
                    outwave=self.spec['WAVE'],
                    normspec_bool=True,
                    )         

               modflux = modflux*self.medflux

               axspec.plot(
                    modwave,modflux,
                    ls='-',
                    lw=0.75,
                    c='blue',
                    alpha=0.15,
                    zorder=-1,
                    )

               axspecr.plot(
                    modwave,(self.spec['FLUX']-modflux)/modflux,
                    ls='-',
                    lw=0.75,
                    c='blue',
                    alpha=0.15,
                    zorder=-1,
                    )


               outarr['chisq_spec'] = np.nansum( 
                    ((self.spec['FLUX']-modflux_bf)/self.spec['EFLUX'])**2.0 
                    )
               outarr['Nspecpix'] = len(self.spec['FLUX'])


               """
          axspec.set_ylabel('Flux')
          axspec.yaxis.tick_right()
          axspec.yaxis.set_label_position('right')
          axspec.set_xlim(self.spec['obs_wave'].min(),self.spec['obs_wave'].max())
          axspec.set_ylim(0.75*specmin,1.2*specmax)

          axspecr.yaxis.tick_right()          
          axspecr.set_xlim(self.spec['obs_wave'].min(),self.spec['obs_wave'].max())
          axspecr.set_ylim(0.75*specrmin,1.25*specrmax)
          axspecr.set_xticks([])

     # def mksed(self,axSED,axMAG,samples):
     #      import star_basis
     #      import photsys
     #      from ccm_curve import ccm_curve

     #      SB = star_basis.StarBasis(
     #           libname='/Users/pcargile/Astro/ckc/ckc_R500.h5',
     #           use_params=['logt','logg','feh'],
     #           n_neighbors=1)

     #      # useful constants
     #      speedoflight = 2.997924e+10
     #      speedoflight_kms = 2.997924e+5
     #      lsun = 3.846e33
     #      pc = 3.085677581467192e18  # in cm
     #      jansky_cgs = 1e-23
     #      # value to go from L_sun to erg/s/cm^2 at 10pc
     #      log_rsun_cgs = np.log10(6.955) + 10.0
     #      log_lsun_cgs = np.log10(lsun)
     #      log4pi = np.log10(4 * np.pi)

     #      WAVE_d = photsys.photsys()
     #      photbands_i = WAVE_d.keys()
     #      photbands = [x for x in photbands_i if x in self.photdata.keys()]
     #      WAVE = {pb:WAVE_d[pb][0] for pb in photbands}
     #      zeropts = {pb:WAVE_d[pb][2] for pb in photbands}
     #      fitsym = {pb:WAVE_d[pb][-2] for pb in photbands}
     #      fitcol = {pb:WAVE_d[pb][-1] for pb in photbands}
     #      filtercurves_i = photsys.filtercurves()
     #      filtercurves = {pb:filtercurves_i[pb] for pb in photbands}

     #      if self.bfdict['[Fe/H]'] >= 0.25:
     #           SEDfeh = 0.25
     #      elif self.bfdict['[Fe/H]'] <= -2.0:
     #           SEDfeh = -2.0
     #      else:
     #           SEDfeh = self.bfdict['[Fe/H]']

     #      if self.bfdict['Teff'] <= 3500.0:
     #           SEDTeff = 3500.0
     #      else:
     #           SEDTeff = self.bfdict['Teff']

     #      if self.bfdict['log(g)'] >= 5.0:
     #           SEDlogg = 5.0
     #      else:
     #           SEDlogg = self.bfdict['log(g)']


     #      spec_w,spec_f,_ = SB.get_star_spectrum(
     #           logt=np.log10(SEDTeff),logg=SEDlogg,feh=SEDfeh)
          
     #      if 'log(A)' in self.bfdict.keys():
     #           lognor = -2.0*self.bfdict['log(A)']+2.0*log_rsun_cgs+log4pi-2.0*np.log10(pc)
     #           nor = 10.0**lognor
     #      else:
     #           to_cgs_i = lsun/(4.0 * np.pi * (pc*self.bfdict['Dist'])**2)
     #           nor = SB.normalize(logr=self.bfdict['log(R)'])*to_cgs_i
     #      spec_f = spec_f*nor
     #      spec_f = spec_f*(speedoflight/((spec_w*1E-8)**2.0))

     #      spec_f = np.nan_to_num(spec_f)
     #      spcond = spec_f > 1e-32
     #      spec_f = spec_f[spcond]
     #      spec_w = spec_w[spcond]

     #      extratio = ccm_curve(spec_w/10.0,self.bfdict['Av']/3.1)                    

     #      axSED.plot(spec_w/(1E+4),np.log10(spec_f/extratio),ls='-',lw=0.5,
     #           alpha=1.0,zorder=-1,c='m')


     #      # do a parameter estimate from the SED alone
     #      SO = SEDopt(
     #           inputphot=self.photdata,
     #           fixedpars={
     #           'Teff':self.bfdict['Teff'],
     #           'Av':self.bfdict['Av'],
     #           'logg':self.bfdict['log(g)'],
     #           'FeH':self.bfdict['[Fe/H]'],
     #           },
     #           returnsed=True,
     #           )
     #      sedpars,sedout = SO()
     #      sedoutkeys = sedout.keys()

     #      # split out data into phot and error dict
     #      initphot = {kk:self.photdata[kk][0] for kk in sedoutkeys if kk in photbands}
     #      initphoterr = {kk:self.photdata[kk][1] for kk in sedoutkeys if kk in photbands}

     #      obswave   = np.array([WAVE[kk] for kk in sedoutkeys])
     #      fitsym    = np.array([fitsym[kk] for kk in sedoutkeys])
     #      fitcol    = np.array([fitcol[kk] for kk in sedoutkeys])
     #      fc        = np.array([filtercurves[kk] for kk in sedoutkeys])
     #      obsmag    = np.array([initphot[kk] for kk in sedoutkeys if kk in photbands])
     #      obsmagerr = np.array([initphoterr[kk] for kk in sedoutkeys if kk in photbands])
     #      modmag    = np.array([sedout[kk] for kk in sedoutkeys])
     #      obsflux_i = np.array([zeropts[kk]*10.0**(initphot[kk]/-2.5) for kk in sedoutkeys if kk in photbands])
     #      obsflux   = [x*(jansky_cgs)*(speedoflight/((lamb*1E-8)**2.0)) for x,lamb in zip(obsflux_i,obswave)]
     #      modflux_i = np.array([zeropts[kk]*10.0**(x/-2.5) for x,kk in zip(modmag,sedoutkeys)])
     #      modflux   = [x*(jansky_cgs)*(speedoflight/((lamb*1E-8)**2.0)) for x,lamb in zip(modflux_i,obswave)]

     #      # plot the observed SED and MAGS
     #      minobsflx = np.inf
     #      maxobsflx = -np.inf
     #      for w,f,mod,s,clr in zip(obswave,obsflux,modflux,fitsym,fitcol):
     #           if np.log10(f) > -30.0:
     #                axSED.scatter(w/1E+4,np.log10(mod),marker=s,c='m',zorder=0,s=100)
     #                axSED.scatter(w/1E+4,np.log10(f),marker=s,c=clr,zorder=1)
     #                if np.log10(f) < minobsflx:
     #                     minobsflx = np.log10(f)
     #                if np.log10(f) > maxobsflx:
     #                     maxobsflx = np.log10(f)

     #      for w,m,me,mod,s,clr in zip(obswave,obsmag,obsmagerr,modmag,fitsym,fitcol):
     #           if np.abs(m-mod)/me > 5.0:
     #                me = np.abs(m-mod)
     #           if (m < 30) & (m > -30):
     #                axMAG.scatter(w/1E+4,mod,marker=s,c='m',zorder=-1,s=100)
     #                axMAG.errorbar(w/1E+4,m,yerr=me,ls='',marker=',',c=clr,zorder=0)
     #                axMAG.scatter(w/1E+4,m,marker=s,c=clr,zorder=1)

     #      # plot filter curves
     #      for fc_i,clr in zip(fc,fitcol):
     #           trans_i = 0.25*fc_i['trans']*(0.9*maxobsflx-1.1*minobsflx)+1.1*minobsflx
     #           axSED.plot(fc_i['wave']/1E+4,trans_i,ls='-',lw=0.5,c=clr,alpha=1.0)




     #      """

     #      try:
     #           if SAMPLES_i['[Fe/H]'] >= 0.25:
     #                SEDfeh_i = 0.25
     #           elif SAMPLES_i['[Fe/H]'] <= -2.0:
     #                SEDfeh_i = -2.0
     #           else:
     #                SEDfeh_i = SAMPLES_i['[Fe/H]']

     #           if SAMPLES_i['Teff'] <= 3500.0:
     #                SEDTeff_i = 3500.0
     #           else:
     #                SEDTeff_i = SAMPLES_i['Teff']

     #           if SAMPLES_i['log(g)'] >= 5.0:
     #                SEDlogg_i = 5.0
     #           else:
     #                SEDlogg_i = SAMPLES_i['log(g)']

     #           spec_w,spec_f,_ = SB.get_star_spectrum(
     #                logt=np.log10(SEDTeff_i),logg=SEDlogg_i,feh=SEDfeh_i)

     #           to_cgs_i = lsun/(4.0 * np.pi * (pc*SAMPLES_i['Dist']*1000.0)**2)
     #           nor = SB.normalize(logr=SAMPLES_i['log(R)'])*to_cgs_i

     #           modphot = self.GM.genphot(
     #                [SAMPLES_i['Teff'],SAMPLES_i['log(g)'],SAMPLES_i['[Fe/H]'],
     #                SAMPLES_i['[a/Fe]'],SAMPLES_i['log(R)'],SAMPLES_i['Dist']*1000.0,
     #                SAMPLES_i['Av'],3.1]
     #                )

     #           for kk in sedout.keys():
     #                moddict[kk].append(modphot[kk])

     #           modphot = np.array([modphot[kk] for kk in sedout.keys() if catfilterarr_i[kk] in spec_out_keys])
                    
     #           spec_f = spec_f*nor
     #           spec_f = spec_f*(speedoflight/((spec_w*1E-8)**2.0))
     #           spec_f = np.nan_to_num(spec_f)
     #           spcond = spec_f > 1e-32
     #           spec_f = spec_f[spcond]
     #           spec_w = spec_w[spcond]     

     #           extratio = ccm_curve(spec_w/10.0,SAMPLES_i['Av']/3.1)
     #           axSED.plot(spec_w/(1E+4),np.log10(spec_f/extratio),ls='-',lw=2.0,alpha=0.1,zorder=-2,c='blue')

     #           for w,m,s in zip(obswave,modphot,fitsym):
     #                axMAG.scatter(w/(1E+4),m,marker=s,c='b')

     #      except ValueError:
     #           pass

     #      """

     #      axSED.set_ylim(1.1*minobsflx,0.9*maxobsflx)

     #      axSED.set_xlim([0.25,6.0])
     #      axSED.set_xscale('log')

     #      axMAG.set_xlim([0.25,6.0])
     #      axMAG.set_xscale('log')

     #      axMAG.set_ylim(axMAG.get_ylim()[::-1])

     #      axSED.set_xticks([0.3,0.5,0.7,1.0,3,5])
     #      axSED.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

     #      axMAG.set_xticks([0.3,0.5,0.7,1,3,5])
     #      axMAG.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

     #      axSED.set_ylabel(r'log(F$_{\lambda}$) [erg s$^{-1}$ cm$^{-2}$]')

     #      axMAG.set_xlabel(r'$\lambda$ [$\mu$m]')
     #      axMAG.set_ylabel('mag')

     #      axSED.yaxis.tick_right()
     #      axMAG.yaxis.tick_right()
     #      axSED.yaxis.set_label_position('right')
     #      axMAG.yaxis.set_label_position('right')
     #      axSED.set_xticklabels([])


if __name__ == '__main__':
     PP = postproc()

     parser = argparse.ArgumentParser()
     parser.add_argument('--starname',    help='starname', type=str, default='HARPS.Archive_18Sco_R32K')
     parser.add_argument('--samplerpath', help='path for output file', type=str, default='./MSoutput/')
     parser.add_argument('--pdfpath',     help='path for PDF file', type=str, default='./MSpdf/')
     args = parser.parse_args()

     PP(**vars(args))