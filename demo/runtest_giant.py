from minesweeper import MINESweeper
MS = MINESweeper.MINESweeper()


"""
MOCK INFO FROM MIST

num_zones    1122.17820115
model_number    1191.17820115
star_age    721602232.975
star_mass    1.99881254373
star_mdot    -9.73084709737e-12
log_dt    4.5801479623
he_core_mass    0.219232269161
c_core_mass    0.0
o_core_mass    0.0
log_L    1.8609889487
log_L_div_Ledd    -1.64465375843
log_LH    1.88234509628
log_LHe    -14.7655574688
log_LZ    -3.61030617422
log_Teff    3.86931485922
log_abs_Lgrav    0.56345756374
log_R    0.714541908088
log_g    3.30983446762
log_surf_z    -2.84502088861
surf_avg_omega    3.39558945652e-05
surf_avg_v_rot    122.385260781
surf_num_c12_div_num_o16    0.436572956681
v_wind_Km_per_s    0.00217525629658
surf_avg_omega_crit    7.47923680262e-05
surf_avg_omega_div_omega_crit    0.454000254562
surf_avg_v_crit    269.570582839
surf_avg_v_div_v_crit    0.454000254562
surf_avg_Lrad_div_Ledd    0.0126980929264
v_div_csound_surf    1.59681803074e-08
surface_h1    0.746262632655
surface_he3    0.000300313342346
surface_he4    0.252008228769
surface_li7    3.87264065359e-19
surface_be9    8.53982544321e-12
surface_b11    0.0
surface_c12    0.000195916803795
surface_c13    7.04035185271e-06
surface_n14    0.000137864807237
surface_o16    0.000598347655433
surface_f19    5.20247749075e-08
surface_ne20    0.000123327878921
surface_na23    3.60901238293e-06
surface_mg24    5.86773695739e-05
surface_si28    6.49560416669e-05
surface_s32    3.11612383641e-05
surface_ca40    6.82107763372e-06
surface_ti48    3.31881338193e-07
surface_fe56    0.000157712930354
log_center_T    7.67725911454
log_center_Rho    4.13256501042
center_degeneracy    1.78932877443
center_omega    0.0055985122091
center_gamma    0.287956251806
mass_conv_core    0.0
center_h1    2.16741453768e-16
center_he4    0.998598357922
center_c12    6.17574069945e-06
center_n14    0.000892464219503
center_o16    7.54839261341e-06
center_ne20    0.000108572820178
center_mg24    5.92415311992e-05
center_si28    6.55710257215e-05
pp    0.173441518211
cno    1.87377144479
tri_alfa    -14.7655574688
burn_c    -12.1613716696
burn_n    -8.54810149422
burn_o    -12.2176661879
c12_c12    -99.0
delta_nu    16.0004442835
delta_Pg    180.647266929
nu_max    203.820473634
acoustic_cutoff    1639.26729229
max_conv_vel_div_csound    0.975475409465
max_gradT_div_grada    24.4193611356
gradT_excess_alpha    0.0
min_Pgas_div_P    0.953769615855
max_L_rad_div_Ledd    0.102154445147
e_thermal    1.17163445671e+49
num_retries    2.0
num_backups    0.0
phase    2.0
initial_Y    0.2511
initial_Z    0.00142857
initial_[Fe/H]    -1.0
initial_[a/Fe]    0.0
initial_vvcrit    0.0
initial_mass    2.0
log_age    8.85829786845
surface_z    0.00142882523351
[Fe/H]    -0.975591159369
EEP    501

"""

priordict = {x:{} for x in ['EEP','initial_mass','initial_[Fe/H]','Dist','Av']}

priordict['EEP']['noninform'] = [0,808]
priordict['initial_mass']['noninform'] = [0.5,4.0]
priordict['initial_[Fe/H]']['noninform'] = [-2.0,0.5]
priordict['Dist']['noninform'] = [0,100.0]
priordict['Av']['noninform'] = [0,0.5]

datadict = {'pars':{},'phot':{}}
datadict['pars']['Teff'] = [10.0**3.86931485922,100.0]
datadict['pars']['log(g)'] = [3.30983446762,0.1]
datadict['pars']['[Fe/H]'] = [-0.975591159369,0.1]

datadict['phot']['SDSS_u']  = [1.40751051456,0.01]
datadict['phot']['SDSS_g']  = [0.225793044276,0.01]
datadict['phot']['SDSS_r']  = [0.184184308357,0.01]
datadict['phot']['SDSS_i']  = [0.240670378513,0.01]
datadict['phot']['SDSS_z']  = [0.297203208513,0.01]
datadict['phot']['2MASS_J']  = [-0.368598495417,0.01]
datadict['phot']['2MASS_H']  = [-0.497012262039,0.01]
datadict['phot']['2MASS_Ks']  = [-0.520924215012,0.01]
datadict['phot']['WISE_W1']  = [-0.539023046188,0.01]
datadict['phot']['WISE_W2']  = [-0.536521796875,0.01]
datadict['phot']['WISE_W3']  = [-0.546512489014,0.01]
datadict['phot']['WISE_W4']  = [-0.561262850456,0.01]

datadict['sampler'] = {}
datadict['sampler']['samplemethod'] = 'rwalk'
datadict['sampler']['npoints'] = 50
datadict['sampler']['samplertype'] = 'single'
datadict['sampler']['flushnum'] = 100

print('TEST MOCK STAR:')
print('----- TRUTH -----')
print('Age: 0.721602232975 Gyr (log(Age) = 8.85829786845)')
print('Init Mass: 2.0 Msol')
print('Dist: 10pc')
print('Av: 0.1')

MS.run(datadict=datadict,priordict=priordict,output='TEST_MIST_giant.dat')
