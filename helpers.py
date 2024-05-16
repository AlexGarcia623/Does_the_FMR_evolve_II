import sys
import os
import time
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import illustris_python as il
import cmasher as cmr

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import stats

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

fs_og = 20
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth'] = 2.25*1.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*1.25
mpl.rcParams['ytick.major.width'] = 1.5*1.25
mpl.rcParams['xtick.minor.width'] = 1.0*1.25
mpl.rcParams['ytick.minor.width'] = 1.0*1.25
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4.5
mpl.rcParams['ytick.minor.size'] = 4.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

BLUE = './Data/'

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02
Zsun   = 1.27E-02

m_star_min = 8.0
m_star_max = 12.0
m_gas_min  = 8.5

WHICH_SIM_TEX = {
    "TNG":r"${\rm TNG}$",
    "ORIGINAL":r"${\rm Illustris}$",
    "EAGLE":r"${\rm EAGLE}$",
    "SIMBA":r"${\rm SIMBA}$"
}

def switch_sim(WHICH_SIM):
    BLUE_DIR = BLUE + WHICH_SIM + "/"
    if (WHICH_SIM.upper() == "TNG"):
        # TNG
        run       = 'L75n1820TNG'
        base      = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/' 
        out_dir   = base 
        snapshots = [99,50,33,25,21,17,13,11,8] # 6,4
        snap2z = {
            99:'z=0',
            50:'z=1',
            33:'z=2',
            25:'z=3',
            21:'z=4',
            17:'z=5',
            13:'z=6',
            11:'z=7',
            8 :'z=8',
            6 :'z=9',
            4 :'z=10',
        }
    elif (WHICH_SIM.upper() == "ORIGINAL"):
        # Illustris
        run       = 'L75n1820FP'
        base      = '/orange/paul.torrey/Illustris/Runs/' + run + '/'
        out_dir   = base
        snapshots = [135,86,68,60,54,49,45,41,38] # 35,32
        snap2z = {
            135:'z=0',
            86 :'z=1',
            68 :'z=2',
            60 :'z=3',
            54 :'z=4',
            49 :'z=5',
            45 :'z=6',
            41 :'z=7',
            38 :'z=8',
            35 :'z=9',
            32 :'z=10',
        }
    elif (WHICH_SIM.upper() == "EAGLE"):
        snapshots = [28,19,15,12,10,8,6,5,4] # 3,2
        snap2z = {
            28:'z=0',
            19:'z=1',
            15:'z=2',
            12:'z=3',
            10:'z=4',
             8:'z=5',
             6:'z=6',
             5:'z=7',
             4:'z=8',
             3:'z=9',
             2:'z=10'
        }
    elif (WHICH_SIM.upper() == "SIMBA"):
        snapshots = [151, 105, 79, 62, 51, 42, 36, 30, 26]
        snap2z = {
            151:'z=0',
            105:'z=1',
             79:'z=2',
             62:'z=3',
             51:'z=4',
             42:'z=5',
             36:'z=6',
             30:'z=7',
             26:'z=8'
        }
    return snapshots, snap2z, BLUE_DIR

def get_all_redshifts(sim,all_z_fit,STARS_OR_GAS='gas',THRESHOLD=-5.00E-01):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    m_star_min = 8.0
    m_star_max = 12.0
    m_gas_min  = 8.5
    
    if sim == "SIMBA": ## Simba is lower res
        m_star_min = 9.0 
        m_gas_min = 9.5
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    redshifts  = []
    redshift   = 0
    
    for snap in snapshots:
        currentDir = BLUE_DIR + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
        sfms_idx = sfmscut(star_mass, SFR, THRESHOLD, m_star_min)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))
        
        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zstar     =     Zstar[desired_mask]
        Zgas      =      Zgas[desired_mask]
        R_gas     =     R_gas[desired_mask]
        R_star    =    R_star[desired_mask]
        
        all_Zgas     += list(Zgas     )
        all_Zstar    += list(Zstar    )
        all_star_mass+= list(star_mass)
        all_gas_mass += list(gas_mass )
        all_SFR      += list(SFR      )
        all_R_gas    += list(R_gas    )
        all_R_star   += list(R_star   )

        redshifts += list( np.ones(len(Zgas)) * redshift )
        
        redshift  += 1
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )
    redshifts = np.array(redshifts     )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    redshifts = redshifts[nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)
    
    Z_use = None
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    return star_mass, SFR, Z_use, redshifts
    

def get_one_redshift(BLUE_DIR,snap,STARS_OR_GAS='gas',THRESHOLD=-5.00E-01):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    
    currentDir = BLUE_DIR + 'snap%s/' %snap

    Zgas      = np.load( currentDir + 'Zgas.npy' )
    Zstar     = np.load( currentDir + 'Zstar.npy' ) 
    star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
    gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
    SFR       = np.load( currentDir + 'SFR.npy' )
    R_gas     = np.load( currentDir + 'R_gas.npy' )
    R_star    = np.load( currentDir + 'R_star.npy' )

    m_star_min = 8.0
    m_star_max = 12.0
    m_gas_min = 8.5
    
    if "SIMBA" in BLUE_DIR:
        m_star_min = 9.0
        m_gas_min = 9.5
    
    sfms_idx = sfmscut(star_mass, SFR, THRESHOLD, m_star_min)

    desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                    (star_mass < 1.00E+01**(m_star_max)) &
                    (gas_mass  > 1.00E+01**(m_gas_min))  &
                    (sfms_idx))

    gas_mass  = gas_mass [desired_mask]
    star_mass = star_mass[desired_mask]
    SFR       = SFR      [desired_mask]
    Zstar     = Zstar    [desired_mask]
    Zgas      = Zgas     [desired_mask]
    R_gas     = R_gas    [desired_mask]
    R_star    = R_star   [desired_mask]

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass

    sSFR[~(sSFR > 0.0)] = 1e-16

    star_mass = star_mass[nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    gas_mass      = np.log10(gas_mass)
    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar
        
    return star_mass, Z_use, SFR
    
def get_z0_alpha(sim,STARS_OR_GAS='gas',function=None):
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    print('Getting z=0 alpha: %s' %sim)
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    snap = snapshots[0]
    
    m_star_min = 8.0
    m_star_max = 12.0
    m_gas_min = 8.5
    
    if sim == "SIMBA":
        m_star_min = 9.0
        m_gas_min = 9.5

    currentDir = BLUE_DIR + 'snap%s/' %snap

    Zgas      = np.load( currentDir + 'Zgas.npy' )
    Zstar     = np.load( currentDir + 'Zstar.npy' ) 
    star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
    gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
    SFR       = np.load( currentDir + 'SFR.npy' )
    R_gas     = np.load( currentDir + 'R_gas.npy' )
    R_star    = np.load( currentDir + 'R_star.npy' )

    sfms_idx = sfmscut(star_mass, SFR, m_star_min = m_star_min)

    desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                    (star_mass < 1.00E+01**(m_star_max)) &
                    (gas_mass  > 1.00E+01**(m_gas_min))  &
                    (sfms_idx))

    gas_mass  =  gas_mass[desired_mask]
    star_mass = star_mass[desired_mask]
    SFR       =       SFR[desired_mask]
    Zstar     =     Zstar[desired_mask]
    Zgas      =      Zgas[desired_mask]
    R_gas     =     R_gas[desired_mask]
    R_star    =    R_star[desired_mask]

    all_Zgas     += list(Zgas     )
    all_Zstar    += list(Zstar    )
    all_star_mass+= list(star_mass)
    all_gas_mass += list(gas_mass )
    all_SFR      += list(SFR      )
    all_R_gas    += list(R_gas    )
    all_R_star   += list(R_star   )
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    alphas = np.linspace(0,1,100)
    all_params = []

    disps = np.ones(len(alphas)) * np.nan
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    for index, alpha in enumerate(alphas):

        mu_fit  = star_mass - alpha*np.log10( SFR )
        
        Z_fit  =  Z_use
        mu_fit = mu_fit
        
        params, cov = curve_fit(function,mu_fit,Z_fit)
        all_params.append(params)
        interp = function(mu_fit, *params)
        
        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 
        
    argmin = np.argmin(disps)

    return round( alphas[argmin], 2 ), *all_params[argmin]

def get_allz_alpha(sim,STARS_OR_GAS='gas',function=None):
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim, False, STARS_OR_GAS)

    alphas = np.linspace(0,1,100)
    all_params = []

    disps = np.ones(len(alphas)) * np.nan
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    for index, alpha in enumerate(alphas):

        mu_fit  = star_mass - alpha*np.log10( SFR )

        Z_fit  =  Z_use
        mu_fit = mu_fit
        
        params, cov = curve_fit(function,mu_fit, Z_fit)
        all_params.append(params)
        interp = function(mu_fit, *params)
        
        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 
        
    argmin = np.argmin(disps)

    return round( alphas[argmin], 2 ), *all_params[argmin]

def get_idv_alpha(star_mass,SFR,Z_use,params,function=None):
    alphas = np.linspace(0,1,100)
    disps = np.ones(len(alphas)) * np.nan

    for index, alpha in enumerate(alphas):
        mu_fit  = star_mass - alpha*np.log10( SFR )

        Z_fit  =  Z_use
        mu_fit = mu_fit
        
        params, cov = curve_fit(function,mu_fit, Z_fit)
        all_params.append(params)
        interp = function(mu_fit, *params)
        
        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 
        
    argmin = np.argmin(disps)

    return round(alphas[argmin], 2)

def plot_fake_MZR(sim,alpha_z0,ax_real,ax_fake,ax_offsets,STARS_OR_GAS='GAS',
                  Type = 'linear'):
    print(sim.upper())
    Type_options = ['linear','fourth-order']
    if Type not in Type_options:
        print('#'*100)
        print('Type not available')
        print('#'*100)
        return
    
    if Type == 'linear':
        func = linear_mu
    elif Type == 'fourth-order':
        func = fourth_order_mu
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
        
    z0_params = None
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                 STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = getMedians(star_mass,Z_true,SFR,
                                                      return_masks=False)
        
        mu = MZR_M_real - alpha_z0 * np.log10(real_SFR)
        
        if index == 0:
            z0_params, cov = curve_fit(func, mu, MZR_Z_real)
            
        ax_real.plot( MZR_M_real, MZR_Z_real, color=f'C{index}',
                      label=r'$z=%s$' %index)
        
        mu = MZR_M_real - alpha_z0 * np.log10(real_SFR)
        MZR_Z_fake = func(mu, *z0_params)
        
        ax_fake.plot( MZR_M_real, MZR_Z_fake, color=f'C{index}',
                      label=r'$z=%s$' %index)
        
        offset = MZR_Z_real - MZR_Z_fake
        print(f'\tMedian Offset: {np.median(offset)}')
        
        ax_offsets.plot( MZR_M_real, offset, color=f'C{index}',
                         label=r'$z=%s$' %index )
        
def plot_fake_MZR_find_alpha(sim,alpha_z0,ax_real,ax_fake,ax_offsets,STARS_OR_GAS='GAS',
                             Type='linear'):
    
    Type_options = ['linear','fourth-order']
    if Type not in Type_options:
        print('#'*100)
        print('Type not available')
        print('#'*100)
        return
    
    if Type == 'linear':
        func = linear_mu
    elif Type == 'fourth-order':
        func = fourth_order_mu
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
        
    z0_MZR = None
    z0_SFMS = None
    z0_params = None
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                 STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = getMedians(star_mass,Z_true,SFR,
                                                      return_masks=False)
        
        mu = MZR_M_real - alpha_z0 * np.log10(real_SFR)

        if index == 0:
            z0_params, cov = curve_fit(func, mu, MZR_Z_real)
            
        ax_real.plot( MZR_M_real, MZR_Z_real, color=f'C{index}',
                      label=r'$z=%s$' %index )
        
        alphas = np.linspace(0,1,100)
        total_offsets = np.empty_like(alphas)
        
        for j, alpha in enumerate(alphas):
            mu = MZR_M_real - alpha * np.log10(real_SFR)
            MZR_Z_fake = func(mu, *z0_params)
            
            ##### WHAT DO WE WANT TO DO ABOUT THIS? #####
            total_offsets[j] = np.sum(np.abs(MZR_Z_real - MZR_Z_fake))
            
        best_alpha = alphas[np.argmin(total_offsets)]
        print(f'{sim.upper()} z={index}: alpha = {best_alpha:.2f}')
        
        mu = MZR_M_real - best_alpha * np.log10(real_SFR)
        MZR_Z_fake = func(mu, *z0_params)
        
        offset = MZR_Z_real - MZR_Z_fake
        print(f'\tMedian Offset: {np.median(offset)}')
        
        ax_fake.plot( MZR_M_real, MZR_Z_fake, color=f'C{index}',
                      label=r'$z=%s$' %index)
        
        ax_offsets.plot( MZR_M_real, offset, color=f'C{index}', label=r'$z=%s$' %index )

def linear_mu(mu, a, b):
    return a * mu + b
        
def fourth_order_mu(mu, a, b, c, d, e):
    return a * mu**4 + b * mu**3 + c * mu**2 + d * mu + e

def ttest(hypothesized_value,measurements,errors):
    l = 16
    # Calculate weighted mean and standard error
    weighted_mean = np.sum(measurements / errors**2) / np.sum(1 / errors**2)
    weighted_std_error = np.sqrt(1 / np.sum(1 / errors**2))

    print(f"\t{'Weighted Mean':<{l}}: {weighted_mean:0.3f}")
    # print(f"\t{'Mean':<{l}}: {np.mean(measurements):0.3f}")
    print(f"\t{'ref val':<{l}}: {hypothesized_value:0.3f}")
    
    # Calculate t-statistic
    t_stat = (weighted_mean - hypothesized_value) / weighted_std_error

    # Degrees of freedom
    degrees_freedom = len(measurements) - 1

    # Calculate p-value (two-tailed)
    p_val = 2 * stats.t.sf(np.abs(t_stat), degrees_freedom)
    print("\t\tWeighted")
    print(f"\t{'T-statistic':<{l}}: {t_stat:0.3f}")
    print(f"\t{'P-value':<{l}}: {p_val:0.3E}")
    print(f"\t{'Reject':<{l}}: {p_val < 0.05}")
    
#     t_stat,p_val = stats.ttest_1samp(measurements, hypothesized_value)
    
#     print("\t\tUnweighted")
#     print(f"\t{'T-statistic':<{l}}: {t_stat:0.3f}")
#     print(f"\t{'P-value':<{l}}: {p_val:0.3E}")
#     print(f"\t{'Reject (p=0.05)':<{l}}: {p_val < 0.05}")
    
def estimate_symmetric_error(lower, upper):
    '''Errors are non symmetric, but not by much. I am just estimating them here'''
    return (lower + upper) / 2
        
def modified_FMR(sim,one_slope=True,STARS_OR_GAS="gas"):
    STARS_OR_GAS = STARS_OR_GAS.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    redshifts  = []
    redshift   = 0
    
    z0_alpha, *_ = get_z0_alpha(sim)
        
    for snap in snapshots:
        currentDir = BLUE_DIR + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
        sfms_idx = sfmscut(star_mass, SFR)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))
        
        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zstar     =     Zstar[desired_mask]
        Zgas      =      Zgas[desired_mask]
        R_gas     =     R_gas[desired_mask]
        R_star    =    R_star[desired_mask]
        
        all_Zgas     += list(Zgas     )
        all_Zstar    += list(Zstar    )
        all_star_mass+= list(star_mass)
        all_gas_mass += list(gas_mass )
        all_SFR      += list(SFR      )
        all_R_gas    += list(R_gas    )
        all_R_star   += list(R_star   )

        redshifts += list( np.ones(len(Zgas)) * redshift )
        
        redshift  += 1
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )
    redshifts = np.array(redshifts     )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    redshifts = redshifts[nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    alphas = np.linspace(0,1,100)

    disps = np.ones(len(alphas)) * np.nan
    
    a_s, b_s = np.ones( len(alphas) ), np.ones( len(alphas) )
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    muCurrent = star_mass - z0_alpha*np.log10( SFR )

    a, b = np.polyfit( muCurrent, Z_use, 1 )

    return z0_alpha, a, b

def modified_FMR_wrong_alpha(sim,one_slope=True,STARS_OR_GAS="gas",
                             alpha=0.0):
    STARS_OR_GAS = STARS_OR_GAS.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    redshifts  = []
    redshift   = 0
            
    for snap in snapshots:
        currentDir = BLUE_DIR + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
        sfms_idx = sfmscut(star_mass, SFR)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))
        
        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zstar     =     Zstar[desired_mask]
        Zgas      =      Zgas[desired_mask]
        R_gas     =     R_gas[desired_mask]
        R_star    =    R_star[desired_mask]
        
        all_Zgas     += list(Zgas     )
        all_Zstar    += list(Zstar    )
        all_star_mass+= list(star_mass)
        all_gas_mass += list(gas_mass )
        all_SFR      += list(SFR      )
        all_R_gas    += list(R_gas    )
        all_R_star   += list(R_star   )

        redshifts += list( np.ones(len(Zgas)) * redshift )
        
        redshift  += 1
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )
    redshifts = np.array(redshifts     )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    redshifts = redshifts[nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    alphas = np.linspace(0,1,100)

    disps = np.ones(len(alphas)) * np.nan
    
    a_s, b_s = np.ones( len(alphas) ), np.ones( len(alphas) )
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    muCurrent = star_mass - alpha*np.log10( SFR )

    a, b = np.polyfit( muCurrent, Z_use, 1 )

    return alpha, a, b
    
def line(data, a, b):
    return a*data + b

def fourth_order( x, a, b, c, d, e ):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

def third_order( x, a, b, c, d ):
    return a + b*x + c*x**2 + d*x**3

def scatter_at_fixed_mu( mu, Z ):
    
    start = np.min(mu)
    end   = np.max(mu)
    
    width = 0.3
    
    current = start
    
    scatter = []
    
    while (current < end):
        
        mask = ( ( mu > current ) &
                 ( mu < current + width) )
        
        scatter.append( np.std( Z[mask] ) * len(Z[mask]) )
        
        current += width
        
    return np.array(scatter)

def sfmscut(m0, sfr0, THRESHOLD=-5.00E-01,m_star_min=8.0):
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) & 
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.0200E+01
    mstp    = 5.0000E-02
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []


    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > THRESHOLD
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00
    
    # Alex added this as a quick bug fix, no idea if it's ``correct''
    nonans = (~(np.isnan(mcs)) &
              ~(np.isnan(rdgs)) &
              ~(np.isnan(rdgs)))
        
    parms, cov = curve_fit(line, mcs[nonans], rdgs[nonans], sigma = rdgstds[nonans])
    mmin    = mbrk
    mmax    = m_star_max
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
        
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > THRESHOLD
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool        

def get_medians(x,y,z,width=0.05,min_samp=15):
    start = np.min(x)
    end   = np.max(x)
    
    xs = np.arange(start,end,width)
    median_y = np.zeros( len(xs) )
    median_z = np.zeros( len(xs) )
    
    for index, current in enumerate(xs):
        mask = ((x > (current)) & (x < (current + width)))
        
        if (len(y[mask]) > min_samp):
            median_y[index] = np.median(y[mask])
            median_z[index] = np.median(z[mask])
        else:
            median_y[index] = np.nan
            median_z[index] = np.nan
        
    nonans = ~(np.isnan(median_y)) & ~(np.isnan(median_z))
    
    xs = xs[nonans] + width
    median_y = median_y[nonans]
    median_z = median_z[nonans]

    return xs, median_y, median_z

if __name__ == "__main__":
    
    print('Hello World!')
