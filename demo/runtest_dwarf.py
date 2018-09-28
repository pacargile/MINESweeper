from minesweeper import MINESweeper
MS = MINESweeper.MINESweeper()

priordict = {x:{} for x in ['EEP','initial_mass','initial_[Fe/H]','Dist','Av']}

priordict['EEP']['noninform'] = [0,808]
priordict['initial_mass']['noninform'] = [0.1,10.0]
priordict['initial_[Fe/H]']['noninform'] = [-1.0,0.5]
priordict['Dist']['noninform'] = [0,1000.0]
priordict['Av']['noninform'] = [0,1.0]

datadict = {}
datadict['pars'] = {}
datadict['pars']['Teff'] = [5770.0,250.0]
datadict['pars']['log(g)'] = [4.44,0.25]
datadict['pars']['[Fe/H]'] = [0.0,0.25]

filterarr = ([
	'PS_g','PS_r','PS_i','PS_z','2MASS_J','2MASS_H','2MASS_Ks',
	'WISE_W1','WISE_W2'])
photarr = ([5.56655009,5.05124561,4.84585131,4.74454356,
	3.79796834,3.42029949,3.33746428,3.2936935,3.31818059])

datadict['phot'] ={ff:[pp,0.05*pp] for ff,pp in zip(filterarr,photarr)}

datadict['sampler'] = {}
datadict['sampler']['samplemethod'] = 'slice'
datadict['sampler']['npoints'] = 250
datadict['sampler']['samplertype'] = 'multi'
datadict['sampler']['flushnum'] = 100

print('TEST MOCK STAR:')
print('----- TRUTH -----')
print('Age: 4.10652483341 Gyr (log(Age) = 9.61347445348)')
print('Init Mass: 1.0 Msol')
print('Dist: 10pc')
print('Av: 0.5')

MS.run(datadict=datadict,priordict=priordict,output='TEST_MIST_dwarf.dat',
	ageweight=False,fastinterp=True)
