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
          
          SAMPLEStab = SAMPLEStab[SAMPLEStab['Prob'] > 0.0]

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
               'Vstellar':   [r'V$_{\bigstar}$ =',' {0:.2f} +{1:.2f}/-{2:.2f} ({3:.2f})\n'],
               'Dist':   [r'Dist =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'Av':     [r'A$_{v}$ =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'log(Age)':       [r'log(Age) =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               # 'initial_[Fe/H]': [r'[Fe/H]$_{i}$ =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               # 'initial_[a/Fe]': [r'[a/Fe]$_{i}$ =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               # 'initial_Mass':   [r'Mass$_{i}$ =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               '[Fe/H]': [r'[Fe/H] =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               '[a/Fe]': [r'[a/Fe] =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'Mass':   [r'Mass =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'log(L)':         [r'log(L) =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               'log(R)':         [r'log(R) =',' {0:.3f} +{1:.3f}/-{2:.3f} ({3:.3f})\n'],
               })

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

          axspec = fig.add_axes([0.45,0.60,0.47,0.25])
          axspecr = fig.add_axes([0.45,0.85,0.47,0.05])

          self.mkspec(axspec,axspecr,SAMPLEStab)

          plt.figtext(0.675,0.35,parstring,fontsize=12)

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
          axspec.set_ylim(0.1*specmin,1.1*specmax)

          axspecr.set_ylabel(r'$\Delta$ Flux / $\sigma$')
          axspecr.yaxis.tick_right()          
          axspecr.yaxis.set_label_position('right')
          axspecr.set_xlim(self.spec['obs_wave'].min(),self.spec['obs_wave'].max())
          axspecr.set_ylim(specrmin,specrmax)
          axspecr.set_xticks([])



if __name__ == '__main__':
     PP = postproc()

     parser = argparse.ArgumentParser()
     parser.add_argument('--starname',    help='starname', type=str, default='HARPS.Archive_18Sco_R32K')
     parser.add_argument('--samplerpath', help='path for output file', type=str, default='./MSoutput/')
     parser.add_argument('--pdfpath',     help='path for PDF file', type=str, default='./MSpdf/')
     args = parser.parse_args()

     PP(**vars(args))