from PayneSw.fitting import fitstar
from astropy.table import Table
import numpy as np
import sys

print('-------- RUNNING MOCK SOLAR DEMO ---------')
print('----- Teff = 5770.0, log(g) = 4.44 -------')
print('-----  [Fe/H] = 0.0, log(L) = 0.0  -------')
print('-----    Av = 0.5, Dist = 10.0     -------')

runspec = False
runphot = True
runmock = True
runiso  = True

print('  ---- Running Spec: {}'.format(runspec))
print('  ---- Running Phot: {}'.format(runphot))
print('  ---- Using Isochrone Prior: {}'.format(runiso))

inputdict = {}

if runspec:
	inputdict['spec'] = {}
	# inputdict['specANNpath'] = '/Users/pcargile/Astro/GITREPOS/PayneSweeper/data/specANN/'

	if runmock:
		demospec = Table.read('demo_spec.fits',format='fits')
		inputdict['spec']['obs_wave'] = demospec['WAVE']
		inputdict['spec']['obs_flux'] = demospec['FLUX']
		# error of SNR = 50
		inputdict['spec']['obs_eflux'] = demospec['FLUX']/20.0
		inputdict['spec']['normspec'] = False
		inputdict['spec']['convertair'] = False
		# set an additional guassian prior on the instrument profile
		# inputdict['priordict']['Inst_R'] = {'gaussian':[32000.0,1000.0]}
	else:
		sunspec = Table.read('ATLAS.Sun_47000.txt.gz',format='ascii')
		sunspec = sunspec[(sunspec['waveobs'] >= 514.5) & (sunspec['waveobs'] <= 532.5)]
		sunspec = sunspec[(sunspec['flux'] != 0.0) & (sunspec['err'] != 0.0)]
		inputdict['spec']['obs_wave'] = sunspec['waveobs']*10.0
		inputdict['spec']['obs_flux'] = sunspec['flux']
		inputdict['spec']['obs_eflux'] = sunspec['err']
		inputdict['spec']['normspec'] = False
		inputdict['spec']['convertair'] = True
		# set an additional guassian prior on the instrument profile
		# inputdict['priordict']['Inst_R'] = {'gaussian':[47000.0,1000.0]}


if runphot:
	inputdict['phot'] = {}
	# inputdict['photANNpath'] = '/Users/pcargile/Astro/GITREPOS/PayneSweeper/data/photANN/'

	if runmock:
		# MOCK PHOT
		phot = ([5.4852947,4.47431828,
			5.56655009,5.05124561,4.84585131,4.74454356,
			3.79796834,3.42029949,#3.33746428,
			3.2936935,3.31818059])
		filterarr = (['Gaia_BP_DR2Rev','Gaia_RP_DR2Rev',
			'PS_g','PS_r','PS_i','PS_z',
			'2MASS_J','2MASS_H',#'2MASS_Ks',
			'WISE_W1','WISE_W2'])

	else:
		phot = ([
			5.03,4.64,4.52,4.51,
			3.67,3.32,3.27,
			3.26,3.28
			])
		filterarr = ([
			'PS_g','PS_r','PS_i','PS_z',
			'2MASS_J','2MASS_H','2MASS_Ks',
			'WISE_W1','WISE_W2'])
	inputdict['phot'] = {fn:[p_i,0.02] for fn,p_i in zip(filterarr,phot)}

# set parameter for sampler
inputdict['sampler'] = {}
inputdict['sampler']['samplemethod'] = 'slice'
inputdict['sampler']['npoints'] = 100
inputdict['sampler']['samplerbounds'] = 'multi'
inputdict['sampler']['flushnum'] = 10
inputdict['sampler']['delta_logz_final'] = 0.1
inputdict['sampler']['bootstrap'] = 0
inputdict['sampler']['slices'] = 100

# set some flat priors for defining the prior volume
inputdict['priordict'] = {}
# inputdict['priordict']['Teff']   = {'uniform':[5000.0,6500.0]}
# inputdict['priordict']['log(g)'] = {'uniform':[4.0,5.0]}
# inputdict['priordict']['[Fe/H]'] = {'uniform':[-0.1,0.1]}
inputdict['priordict']['[a/Fe]'] = {'uniform':[-0.1,0.1]}
inputdict['priordict']['Vrad']   = {'uniform':[-5.0,5.0]}
inputdict['priordict']['Vrot']   = {'uniform':[0.0,5.0]}
inputdict['priordict']['Inst_R'] = {'uniform':[25000.0,55000.0]}

inputdict['priordict']['Dist']   = {'uniform':[1.0,20.0]}
# inputdict['priordict']['log(R)'] = {'uniform':[-0.5,1.5]}
inputdict['priordict']['Av']     = {'uniform':[0.0,1.0]}
# inputdict['priordict']['log(A)'] = {'uniform':[-0.1,0.1]}

inputdict['priordict']['EEP'] = {'uniform':[200,808]}

inputdict['isochrone_prior'] = runiso

inputdict['output'] = 'demoout.dat'


FS = fitstar.FitPayne()
print('---------------')
if 'phot' in inputdict.keys():
	print('    PHOT:')
	for kk in inputdict['phot'].keys():
	        print('       {0} = {1} +/- {2}'.format(kk,inputdict['phot'][kk][0],inputdict['phot'][kk][1]))
if 'spec' in inputdict.keys():
	print('    Median Spec Flux: ')
	print('       {0}'.format(np.median(inputdict['spec']['obs_flux'])))
	print('    Median Spec Err_Flux:')
	print('       {0}'.format(np.median(inputdict['spec']['obs_eflux'])))

if 'priordict' in inputdict.keys():
	print('    PRIORS:')
	for kk in inputdict['priordict'].keys():
		if kk == 'blaze_coeff':
			pass
		else:
			for kk2 in inputdict['priordict'][kk].keys():
				if kk2 == 'uniform':
					print('       {0}: min={1} max={2}'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
				if kk2 == 'gaussian':
					print('       {0}: N({1},{2})'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))

print('--------------')

sys.stdout.flush()
result = FS.run(inputdict=inputdict)
sys.stdout.flush()


