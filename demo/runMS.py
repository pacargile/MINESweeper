import numpy as np
from minesweeper import fitstar
import json

from astropy.table import Table

# Define location of models needed for fitting
SPECNN = '/Users/pcargile/Astro/ThePayne/YSdata/YSTANN.h5'
PHOTNN = '/Users/pcargile/Astro/ThePayne/SED/VARRV/'
MISTISO = '/Users/pcargile/Astro/MIST/MIST_v2.0_spot/MIST_2.0_spot_EEPtrk_small.h5'

# Read in demo info
inspec = Table.read('./spectra/HARPS.Archive_18Sco_R32K.fits',format='fits')
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

# init the input dict
inputdict = {}

######################
# build spectrum input
inputdict['spec'] = {}

# define NN information
inputdict['specANNpath'] = SPECNN
inputdict['NNtype'] = 'YST1'

# input in spec data
inputdict['spec']['obs_wave']  = inspec['waveobs']
inputdict['spec']['obs_flux']  = inspec['flux']
# inputdict['spec']['obs_eflux'] = inspec['EFLUX']
# add noise floor to spectrum, like 1% error
inputdict['spec']['obs_eflux'] = np.array(
    [np.sqrt(x**2.0 + (0.01*y)**2.0) for x,y in zip(inspec['err'],inspec['flux'])])

# divide data by median -> ThePayne/MINESweeper works on median flux
medflux = np.nanmedian(inputdict['spec']['obs_flux'])
medflux_err = np.nanmedian(inputdict['spec']['obs_eflux'])
inputdict['spec']['obs_flux']  = inputdict['spec']['obs_flux']/medflux
inputdict['spec']['obs_eflux'] = inputdict['spec']['obs_eflux']/medflux

# set switch to tell MS to fit blaze function Cheb. poly
inputdict['spec']['modpoly'] = True

# set switch to tell MS that data is in air or vaccum
inputdict['spec']['convertair'] = True


########################
# build photometry input
inputdict['phot'] = {}

# pull phot data from input file
for inphot_i in inphot:
    filt = inphot_i['filter']
    phot = inphot_i['phot']
    err  = inphot_i['err']

    if np.isfinite(phot):
        # Add a noise floor for photometry
        if filt.split('_')[0] == 'GaiaEDR3':
            err = np.sqrt(err**2.0 + 0.01**2.0)
        elif filt.split('_')[0] == '2MASS':
            err = np.sqrt(err**2.0 + 0.05**2.0)
        elif filt.split('_')[0] == 'WISE':
            err = np.sqrt(err**2.0 + 0.05**2.0)
        else:
            pass

        inputdict['phot'][filt] = [phot,err]

# define NN information
inputdict['photANNpath'] = PHOTNN

# Fit R_V reddening vector (complelety not tested)
inputdict['Rvfree'] = False


################
# MIST parameters

# define MIST isochrone file
inputdict['MISTpath'] = MISTISO
inputdict['isochrone_prior'] = True
inputdict['ageweight'] = True

#####################
# define sampler info

# set parameter for sampler
inputdict['sampler'] = {}

inputdict['sampler']['samplertype']      = insamp_i['samplertype']
inputdict['sampler']['samplemethod']     = insamp_i['samplemethod']
inputdict['sampler']['npoints']          = insamp_i['npoints']
inputdict['sampler']['samplerbounds']    = insamp_i['samplerbounds']
inputdict['sampler']['flushnum']         = insamp_i['flushnum']
inputdict['sampler']['delta_logz_final'] = insamp_i['delta_logz_final']
inputdict['sampler']['walks']            = insamp_i['walks']
inputdict['sampler']['maxcall']          = insamp_i['maxcall']
inputdict['sampler']['maxiter']          = insamp_i['maxiter']

############## Priors ################
inputdict['priordict'] = {}

# prior which change prior volume
# 
# pv_uniform
# pv_gaussian
# pv_tgaussian
# pv_exp
# pv_texp
# pv_loguniform
#

inputdict['priordict']['EEP'] = {'pv_uniform':[200,808]}
inputdict['priordict']['initial_Mass'] = {'pv_uniform':[0.5,1.25]}
inputdict['priordict']['initial_[Fe/H]'] = {'pv_uniform':[-4.0,0.0]}
inputdict['priordict']['initial_[a/Fe]'] = {'pv_uniform':[-0.2,0.6]}

inputdict['priordict']['Dist']   = {'pv_uniform':[1.0,100.0]}
inputdict['priordict']['Av']     = {'pv_uniform':[0.0,0.1]}

inputdict['priordict']['Vrad'] = {'pv_uniform':[-5.0,5.0]}
# inputdict['priordict']['Vrot'] = ({'pv_tgaussian':[0.0,7.0,0.0,2.0],})
inputdict['priordict']['Vrot'] = ({'pv_uniform':[0.0,10.0]})

# fixed sampling parameters 
inputdict['priordict']['Inst_R'] = ({'fixed':35000.0})

# Priors for predicted a posteriori parameters
#
# uniform
# gaussian
# 
inputdict['priordict']['Age'] = {'uniform':[1.0,14.0]}
inputdict['priordict']['Parallax'] =  ({'gaussian':indata_i['Parallax']})

# Advanced Priors
# 
# IMF    -> Init Mass Function
# VROT   -> Physically Inspired Rotation Model
# GAL    -> Galactic Density Model
# GALAGE -> Galactic Age Model
# VTOT   -> Max total Velocity < 600 km/s
#
inputdict['priordict']['IMF'] = {'IMF_type':'Kroupa'}

# If fitting blaze function, set priors on poly coefficients
coeffarr = [[1.0,0.5],[0.0,0.075],[0.0,0.025],[0.0,0.01]]
inputdict['priordict']['blaze_coeff'] = coeffarr

# Define output file name
inputdict['output'] = 'MSoutput/HARPS.Archive_18Sco_R32K.dat'

# print out info for user log
print('--- Photometry in Fit ---')
for kk in inputdict['phot'].keys():
    print('{0} = {1} +/- {2}'.format(
        kk,inputdict['phot'][kk][0],inputdict['phot'][kk][1]))

print('--- Spectral Information in Fit ---')
print('Wavelength Range = {0} -- {1}'.format(
    inputdict['spec']['obs_wave'].min(),inputdict['spec']['obs_wave'].max()))
print('Median Flux = {0}'.format(medflux))
print('Median Flux Error = {0}'.format(medflux_err))
print('SNR = {0}'.format(medflux/medflux_err))

# Init the fitter
FS = fitstar.FitMS()

# Now run fit
results = FS.run(inputdict=inputdict)

