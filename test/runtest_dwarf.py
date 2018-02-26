from minesweeper import MINESweeper
MS = MINESweeper.MINESweeper()


"""
MOCK INFO FROM MIST

num_zones    767.0
model_number    737.979535893
star_age    4106524833.41
star_mass    0.999859536783
star_mdot    -4.16635234187e-14
log_dt    8.31000907579
he_core_mass    0.0
c_core_mass    0.0
o_core_mass    0.0
log_L    0.0256371271847
log_L_div_Ledd    -3.72189134786
log_LH    0.025576860078
log_LHe    -42.6341145891
log_LZ    -5.10567293419
log_Teff    3.76624868855
log_abs_Lgrav    -3.85587356068
log_R    0.00299833867054
log_g    4.43208853353
log_surf_z    -1.88537942766
surf_avg_omega    0.0
surf_avg_v_rot    0.0
surf_num_c12_div_num_o16    0.54489719669
v_wind_Km_per_s    1.87733354781e-05
surf_avg_omega_crit    0.0
surf_avg_omega_div_omega_crit    0.0
surf_avg_v_crit    0.0
surf_avg_v_div_v_crit    0.0
surf_avg_Lrad_div_Ledd    0.0
v_div_csound_surf    3.2271454533e-13
surface_h1    0.743444489397
surface_he3    6.19038887595e-05
surface_he4    0.243473316146
surface_li7    9.65011932941e-10
surface_be9    1.54934548915e-10
surface_b11    0.0
surface_c12    0.00228905209163
surface_c13    2.77862004741e-05
surface_n14    0.000676014220624
surface_o16    0.00560118399224
surface_f19    4.85780356317e-07
surface_ne20    0.00111647497651
surface_na23    2.81356732857e-05
surface_mg24    0.000531186124269
surface_si28    0.000588024797396
surface_s32    0.00028209201813
surface_ca40    6.17488795866e-05
surface_ti48    3.0044080847e-06
surface_fe56    0.00142772114153
log_center_T    7.19369207871
log_center_Rho    2.16499368112
center_degeneracy    -1.53131564611
center_omega    0.0
center_gamma    0.105244076803
mass_conv_core    0.0
center_h1    0.37260130203
center_he4    0.612115477428
center_c12    7.17182558584e-06
center_n14    0.00381860893738
center_o16    0.00588871752144
center_ne20    0.00129590109002
center_mg24    0.000616569646386
center_si28    0.000682544637038
pp    0.0228682264365
cno    -2.18087338655
tri_alfa    -42.6341145891
burn_c    -20.4260003848
burn_n    -17.4443966588
burn_o    -19.3796100478
c12_c12    -99.0
delta_nu    140.461246899
delta_Pg    0.0
nu_max    3041.10799556
acoustic_cutoff    27772.6952615
max_conv_vel_div_csound    0.315057533653
max_gradT_div_grada    2.72244905266
gradT_excess_alpha    0.0
min_Pgas_div_P    0.998836882771
max_L_rad_div_Ledd    0.00156382768201
e_thermal    5.15681316471e+48
num_retries    0.0
num_backups    0.0
phase    0.0
initial_Y    0.2703
initial_Z    0.0142857
initial_[Fe/H]    0.0
initial_[a/Fe]    0.0
initial_vvcrit    0.0
initial_mass    1.0
log_age    9.61347445348
surface_z    0.0130202874512
[Fe/H]    -0.0143065492335
EEP    350
"""

priordict = {x:{} for x in ['EEP','initial_mass','initial_[Fe/H]','Dist','Av']}

priordict['EEP']['noninform'] = [0,808]
priordict['initial_mass']['noninform'] = [0.5,2.0]
priordict['initial_[Fe/H]']['noninform'] = [-1.0,0.5]
priordict['Dist']['noninform'] = [0,100.0]
priordict['Av']['noninform'] = [0,0.5]

datadict = {'pars':{},'phot':{}}
datadict['pars']['Teff'] = [10.0**3.76624868855,250.0]
datadict['pars']['log(g)'] = [4.43208853353,0.25]
datadict['pars']['[Fe/H]'] = [-0.01430655,0.25]

datadict['phot']['SDSS_u']  = [6.391256141,0.05]
datadict['phot']['SDSS_g']  = [5.103466557,0.05]
datadict['phot']['SDSS_r']  = [4.655638622,0.05]
datadict['phot']['SDSS_i']  = [4.528017150,0.05]
datadict['phot']['SDSS_z']  = [4.529033499,0.05]
datadict['phot']['2MASS_J'] = [3.644908236,0.05]
datadict['phot']['2MASS_H'] = [3.312467383,0.05]
datadict['phot']['2MASS_Ks']= [3.271397995,0.05]
datadict['phot']['WISE_W1'] = [3.260185407,0.05]
datadict['phot']['WISE_W2'] = [3.265715050,0.05]
datadict['phot']['WISE_W3'] = [3.222959088,0.05]
datadict['phot']['WISE_W4'] = [3.236377047,0.05]

datadict['sampler'] = {}
datadict['sampler']['samplemethod'] = 'rwalk'
datadict['sampler']['npoints'] = 50
datadict['sampler']['samplertype'] = 'single'
datadict['sampler']['flushnum'] = 100

print('TEST MOCK STAR:')
print('----- TRUTH -----')
print('Age: 4.10652483341 Gyr (log(Age) = 9.61347445348)')
print('Init Mass: 1.0 Msol')
print('Dist: 10pc')
print('Av: 0.1')

MS.run(datadict=datadict,priordict=priordict,output='TEST_MIST_dwarf.dat')