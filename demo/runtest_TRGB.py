from minesweeper import MINESweeper
MS = MINESweeper.MINESweeper()

"""
MOCK INFO FROM MIST

num_zones    1922.0
model_number    14091.0
star_age    3359681546.31
star_mass    1.47172615002
star_mdot    -1.08072105197e-08
log_dt    2.51485682174
he_core_mass    0.464375150814
c_core_mass    0.0
o_core_mass    0.0
log_L    3.37447248413
log_L_div_Ledd    -3.3211300156
log_LH    3.36948769411
log_LHe    1.43414008761
log_LZ    -0.698487776624
log_Teff    3.49332456054
log_abs_Lgrav    -0.32260905651
log_R    2.22326427317
log_g    0.159444677821
log_surf_z    -1.59215674072
surf_avg_omega    3.39226912097e-09
surf_avg_v_rot    0.38717248482
surf_num_c12_div_num_o16    0.324729492431
v_wind_Km_per_s    5.68871581694e-05
surf_avg_omega_crit    3.61805466661e-07
surf_avg_omega_div_omega_crit    0.00937657854442
surf_avg_v_crit    41.2923498635
surf_avg_v_div_v_crit    0.00937657854442
surf_avg_Lrad_div_Ledd    0.00323506290657
v_div_csound_surf    4.38613188051e-07
surface_h1    0.672728831756
surface_he3    0.000252564371199
surface_he4    0.301441977501
surface_li7    1.01749170544e-12
surface_be9    1.77483723065e-11
surface_b11    0.0
surface_c12    0.00263365396335
surface_c13    0.000340589223128
surface_n14    0.0030908598867
surface_o16    0.0108137348151
surface_f19    9.58922062747e-07
surface_ne20    0.00219541248938
surface_na23    5.71946726141e-05
surface_mg24    0.00104451317227
surface_si28    0.00115627953825
surface_s32    0.000554699784623
surface_ca40    0.000121421692235
surface_ti48    5.90780458288e-06
surface_fe56    0.00280744068889
log_center_T    7.87850257109
log_center_Rho    5.92527139381
center_degeneracy    19.8297923295
center_omega    0.0293265345043
center_gamma    0.739419477801
mass_conv_core    0.0
center_h1    3.29646559633e-22
center_he4    0.974652740383
center_c12    8.09684727027e-05
center_n14    0.0155457287435
center_o16    0.000829504327289
center_ne20    0.00223447118856
center_mg24    0.00107161557925
center_si28    0.00118017574628
pp    -0.254492397408
cno    3.36938445225
tri_alfa    1.43414008761
burn_c    -1.92020018697
burn_n    -2.38236454953
burn_o    -3.44289787303
c12_c12    -99.0
delta_nu    0.0940616936423
delta_Pg    28.7670189134
nu_max    0.222253729639
acoustic_cutoff    1.93529992257
max_conv_vel_div_csound    0.599243459985
max_gradT_div_grada    7.20198605067
gradT_excess_alpha    0.0
min_Pgas_div_P    0.773700515375
max_L_rad_div_Ledd    0.256420727541
e_thermal    5.34299145892e+48
num_retries    0.0
num_backups    0.0
phase    3.0
initial_Y    0.2869
initial_Z    0.0254039
initial_[Fe/H]    0.25
initial_[a/Fe]    0.0
initial_vvcrit    0.0
initial_mass    1.5
log_age    9.52629811393
surface_z    0.0255766263724
[Fe/H]    0.322324643372
EEP    605

"""

priordict = {x:{} for x in ['EEP','initial_mass','initial_[Fe/H]','Dist','Av']}

priordict['EEP']['noninform'] = [0,808]
priordict['initial_mass']['noninform'] = [0.5,2.0]
priordict['initial_[Fe/H]']['noninform'] = [-1.0,0.5]
priordict['Dist']['noninform'] = [0,100.0]
priordict['Av']['noninform'] = [0,0.5]

datadict = {}

datadict['pars'] = {}
datadict['pars']['Teff'] = [10.0**3.49332456054,100.0]
datadict['pars']['log(g)'] = [0.159444677821,0.1]
datadict['pars']['[Fe/H]'] = [0.322324643372,0.1]

datadict['phot'] = {}
datadict['phot']['SDSS_u']    = [4.25767764206,0.01]
datadict['phot']['SDSS_g']    = [2.2464958107,0.01]
datadict['phot']['SDSS_r']    = [0.738126364901,0.01]
datadict['phot']['SDSS_i']    = [-1.74844876175,0.01]
datadict['phot']['SDSS_z']    = [-23.0787705505,0.01]
datadict['phot']['2MASS_J']   = [-5.49827245121,0.01]
datadict['phot']['2MASS_H']   = [-6.56436386947,0.01]
datadict['phot']['2MASS_Ks']  = [-6.90378013496,0.01]
datadict['phot']['WISE_W1']   = [-6.9270743454,0.01]
datadict['phot']['WISE_W2']   = [-6.78366080169,0.01]
datadict['phot']['WISE_W3']   = [-7.033449086,0.01]
datadict['phot']['WISE_W4']   = [-7.17153612022,0.01]

datadict['sampler'] = {}
datadict['sampler']['samplemethod'] = 'rwalk'
datadict['sampler']['npoints'] = 50
datadict['sampler']['samplertype'] = 'single'
datadict['sampler']['flushnum'] = 100

print('TEST MOCK STAR:')
print('----- TRUTH -----')
print('Age: 3.35968154631 Gyr (log(Age) = 9.52629811393)')
print('Init Mass: 1.5 Msol')
print('Dist: 10pc')
print('Av: 0.1')

MS.run(datadict=datadict,priordict=priordict,output='TEST_MIST_TRGB.dat')