import os
import sys
import time
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import illustris_python as il

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

sys.path.insert(1,'./Data/')
from additional_data import *

from helpers import (WHICH_SIM_TEX, switch_sim, get_one_redshift,
                     get_z0_alpha, sfmscut, modified_FMR)

BLUE = './Data/'

WHICH_SIM    = "eagle".upper() 
STARS_OR_GAS = "gas".upper() # stars or gas

BLUE_DIR = BLUE + WHICH_SIM + "/"

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

def do(ax,sim,c,all_z_fit,STARS_OR_GAS='gas'):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_offsets = []
    means       = []
    
    z0_alpha, z0_a, z0_b = 0,0,0
    
    for snap in snapshots:
        
        star_mass, Z_use, SFR = get_one_redshift(BLUE_DIR,snap,
                                                 STARS_OR_GAS=STARS_OR_GAS)
        
        if snap2z[snap] == 'z=0':
            if (all_z_fit):
                z0_alpha, z0_a, z0_b = modified_FMR(sim,one_slope=True)
            else:
                alphas = np.linspace(0,1,100)
                disp   = np.zeros( len(alphas) )
                a_s    = np.zeros( len(alphas) )
                b_s    = np.zeros( len(alphas) )

                for index, alpha in enumerate(alphas):

                    muCurrent = star_mass - alpha*np.log10(SFR) 

                    popt = np.polyfit(muCurrent, Z_use, 1)

                    a_s[index], b_s[index] = popt

                    interp = np.polyval( popt, muCurrent )

                    disp[index] = np.std( np.abs(Z_use) - np.abs(interp) )

                argmin = np.argmin(disp)
                
                z0_alpha = round( alphas[argmin], 2 )
                z0_a     = a_s[argmin]
                z0_b     = b_s[argmin]
            
        mu = star_mass - z0_alpha * np.log10(SFR)
        
        z0_FMR_Z_predictions = z0_a * mu + z0_b
        
        offsets = Z_use - z0_FMR_Z_predictions
        
        all_offsets.append( offsets )
        means.append( np.median(offsets) )
        
    bp = ax.boxplot( all_offsets, patch_artist=True,
                     whiskerprops = dict(color = c,alpha=0.5),
                     capprops     = dict(color = c,alpha=0.5),
                     boxprops     = dict(facecolor = 'white',color=c,alpha=0.5),
                     flierprops   = dict(marker='+', alpha=0.25,markersize=2.5,markerfacecolor=c,
                                         markeredgecolor=c),
                     widths       = np.ones(len(means)) * 0.25)
    for median in bp['medians']:
        median.set_color(color)
        
    if not all_z_fit:
        for index, coords in enumerate(Langeroodi23):
            x = coords[0]
            y = coords[1]
            x_err_up   = Langeroodi23_up[index][0] - x
            x_err_down = x - Langeroodi23_down[index][0]
            y_err_up   = Langeroodi23_yup[index][1] - y
            y_err_down = y - Langeroodi23_ydown[index][1]

            x_err = np.array([ [x_err_down, x_err_up] ]).T
            y_err = np.array([ [y_err_down, y_err_up] ]).T
            if (index == 0):
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='purple', marker='s', markersize=8,
                           label = r'${\rm Langeroodi\;\&\;Hjorth\;(2023)}$' )
            else:
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='purple', marker='s', markersize=8 )

        for index, coords in enumerate(Curti23):
            x = coords[0]
            y = coords[1]
            x_err_up   = Curti23_up[index][0] - x
            x_err_down = x - Curti23_down[index][0]
            y_err_up   = Curti23_yup[index][1] - y
            y_err_down = y - Curti23_ydown[index][1]

            x_err = np.array([ [x_err_down, x_err_up] ]).T
            y_err = np.array([ [y_err_down, y_err_up] ]).T
            if (index == 0):
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='violet', marker='^', markersize=8,
                           label = r'${\rm Curti+(2023)}$' )
            else:
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='violet', marker='^', markersize=8 )

        for index, coords in enumerate(Nakajima23_C20):
            x = coords[0]
            y = coords[1]
            x_err_up   = Nakajima23_C20_up[index][0] - x
            x_err_down = x - Nakajima23_C20_down[index][0]
            y_err_up   = Nakajima23_C20_yup[index][1] - y
            y_err_down = y - Nakajima23_C20_ydown[index][1]

            x_err = np.array([ [x_err_down, x_err_up] ]).T
            y_err = np.array([ [y_err_down, y_err_up] ]).T
            if (index == 0):
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='goldenrod', marker='*', markersize=10,
                           label = r'${\rm Nakajima+(2023; C20)}$' )
            else:
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='goldenrod', marker='*', markersize=10 )

        for index, coords in enumerate(Nakajima23_AM13):
            x = Nakajima23_C20[index][0]#coords[0]
            y = coords[1]
            x_err_up   = Nakajima23_C20_up[index][0] - x
            x_err_down = x - Nakajima23_C20_down[index][0]
            y_err_up   = Nakajima23_AM13_yup[index][1] - y
            y_err_down = y - Nakajima23_AM13_ydown[index][1]

            x_err = np.array([ [x_err_down, x_err_up] ]).T
            y_err = np.array([ [y_err_down, y_err_up] ]).T
            if (index == 0):
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='navy', marker='o', markersize=8,
                           label = r'${\rm Nakajima+(2023; AM13)}$' )
            else:
                ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='navy', marker='o', markersize=8 )
            
    ax.set_xticks( np.arange(1,10) )     
    ax.set_xticklabels( np.arange(0,9) )
    
    redshifts = np.arange(0,9) + 1
        
    popt   = np.polyfit( redshifts, means, 1 )
    interp = np.polyval( popt, redshifts )
    
    ax.plot( redshifts, interp, color='k', lw=2.5, linestyle='--' )
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax.text( 0.05, 0.85, WHICH_SIM_TEX[sim], transform=ax.transAxes, color=color )

sims   = ['ORIGINAL','TNG','EAGLE']
cols   = ['C1','C2','C0']

savedir = './Figures (pdfs)/'

for all_z_fit in [False, True]:
    plt.clf()
    fig,axs = plt.subplots(3,1,figsize=(8,13),sharex=True, sharey=True)
    
    for index, sim in enumerate(sims):
        ax = axs[index]
        color = cols[index]#'k'#'C' + str(index)
        do(ax, sim, color, all_z_fit)

    ymin, ymax = axs[0].get_ylim()
    axs[0].set_ylim( -1.1, 1.1 )

    axs[0].set_xlim(0,10)

    if not all_z_fit:
        leg = axs[1].legend( loc='upper right', frameon=False, fontsize=18,
                             handlelength=0, labelspacing=0.05 )
        colors = ['purple','violet','goldenrod','navy']
        for index, text in enumerate(leg.get_texts()):
            text.set_color(colors[index])

    axs[1].set_ylabel( r'$\log {\rm (O/H)} - \log{\rm (O/H)}_{{\rm FMR}}$' )

    axs[2].set_xlabel( r'${\rm Redshift}$' )

    if all_z_fit:
        axs[0].set_title(r'${\rm All}~z~{\rm fit}$')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    Fig_num = 2
    if (all_z_fit):
        Fig_num = 4

    plt.savefig( savedir + 'Figure%s' %Fig_num + '.pdf', bbox_inches='tight' )