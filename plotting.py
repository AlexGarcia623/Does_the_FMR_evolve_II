import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import cmasher as cmr

from helpers import (
    WHICH_SIM_TEX, get_z0_alpha, get_all_redshifts,
    get_one_redshift, get_medians, switch_sim, get_allz_alpha,
    get_idv_alpha
)

def fourth_order(mu, a, b, c, d, e):
    return a * mu**4 + b * mu**3 + c * mu**2 + d * mu + e

def linear(mu, a, b):
    return a * mu + b

def make_FMR_fig(sim,all_z_fit,STARS_OR_GAS="gas",savedir="./",
                 function = fourth_order,verbose=True):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,all_z_fit)

    min_alpha, *params = get_z0_alpha(sim, function=function)
    best_mu = star_mass - min_alpha*np.log10(SFR)
    if (all_z_fit):
        params, cov = curve_fit(function, best_mu, Z_use)

    if verbose:
        print(f'Simulation: {sim}')
        print(f'All_redshift: {all_z_fit}')
        print(f'Params: {params}')

    plot_mu = np.linspace(np.min(best_mu),np.max(best_mu),100)
    best_line = function(plot_mu, *params)

    unique, n_gal = np.unique(redshifts, return_counts=True)

    CMAP_TO_USE = cmr.get_sub_cmap('cmr.guppy_r', 0.0, 1.0, N=len(redshifts))

    plt.clf()
    fig = plt.figure(figsize=(30,20))

    gs  = gridspec.GridSpec(4, 7, width_ratios = [0.66,0.66,0.66,0.35,1,1,1],
                             height_ratios = [1,1,1,0.4], wspace = 0.0, hspace=0.0)

    axBig = fig.add_subplot( gs[:,:3] )

    Hist1, xedges, yedges = np.histogram2d(best_mu,Z_use,weights=redshifts,bins=(100,100))
    Hist2, _     , _      = np.histogram2d(best_mu,Z_use,bins=[xedges,yedges])

    Hist1 = np.transpose(Hist1)
    Hist2 = np.transpose(Hist2)

    hist = Hist1/Hist2

    mappable = axBig.pcolormesh( 
        xedges, yedges, hist, vmin = 0, vmax = 8, cmap=CMAP_TO_USE, rasterized=True
    )

    cbar = plt.colorbar( mappable, label=r"${\rm Redshift}$", orientation='horizontal' )
    cbar.ax.set_xticks(np.arange(0,9))
    cbar.ax.set_xticklabels(np.arange(0,9))

    axBig.plot( plot_mu, best_line, color='k', lw=8.0 )

    if (STARS_OR_GAS == "GAS"):
        plt.ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$')
    elif (STARS_OR_GAS == "STARS"):
        plt.ylabel(r'$\log(Z_* [Z_\odot])$')
    plt.xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))

    axBig.text( 0.05, 0.9, "%s" %WHICH_SIM_TEX[sim], transform=axBig.transAxes )

    Hist1, xedges, yedges = np.histogram2d(best_mu,Z_use,bins=(100,100))
    Hist2, _     , _      = np.histogram2d(best_mu,Z_use,bins=[xedges,yedges])

    percentage = 0.01
    xmin, xmax = np.min(best_mu)*(1-percentage), np.max(best_mu)*(1+percentage)
    ymin, ymax = np.min(Z_use)  *(1-percentage), np.max(Z_use)  *(1+percentage)

    axBig.set_xlim(xmin,xmax)
    axBig.set_ylim(ymin,ymax)

    Hist1 = np.transpose(Hist1)
    Hist2 = np.transpose(Hist2)

    hist = Hist1/Hist2

    ax_x = 0
    ax_y = 4

    axInvis = fig.add_subplot( gs[:,3] )
    axInvis.set_visible(False)

    small_axs = []
    ylab_flag = True

    for index, time in enumerate(unique):
        ax = fig.add_subplot( gs[ax_x, ax_y] )

        small_axs.append(ax)

        if (ylab_flag):
            if (STARS_OR_GAS == "GAS"):
                ax.set_ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$',fontsize=36 )
            elif (STARS_OR_GAS == "STARS"):
                ax.set_ylabel(r'$\log(Z_* [Z_\odot])$',fontsize=36 )

            ylab_flag = False

        if (ax_x == 2):
            ax.set_xlabel( r'$\mu_{%s}$' %(min_alpha),
                           fontsize=36 )

        if (ax_y == 5 or ax_y == 6):
            ax.set_yticklabels([])
        if (ax_y == 0 or ax_y == 1):
            ax.set_xticklabels([])

        ax_y += 1
        if (ax_y == 7):
            ax_y = 4
            ax_x += 1
            ylab_flag = True

        mask = (redshifts == time)

        ax.pcolormesh( xedges, yedges, hist, alpha=0.25, vmin = 0, vmax = 1.5,
                       cmap=plt.cm.Greys, rasterized=True )

        current_mu    =   best_mu[mask]
        current_Z     =     Z_use[mask]

        Hist1, current_x, current_y = np.histogram2d(current_mu,current_Z,bins=[xedges, yedges])
        Hist2, _        , _         = np.histogram2d(current_mu,current_Z,bins=[current_x,current_y])

        Hist1 = np.transpose(Hist1)
        Hist2 = np.transpose(Hist2)

        current_hist = Hist1/Hist2

        vmin = 1 - time
        vmax = 9 - time

        ax.pcolormesh( 
            current_x, current_y, current_hist, vmin = vmin, vmax = vmax,
            cmap=CMAP_TO_USE, rasterized=True 
        )

        ax.plot( plot_mu, best_line, color='k', lw=6 )

        ax.text( 0.65, 0.1, r"$z = %s$" %int(time), transform=plt.gca().transAxes )

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    if function == linear:
        _a_, _b_ = params
        x=9.25
        l2 = np.array((x, _a_*x+_b_+0.05))
        rotation = np.arctan(_a_)
        rotation = np.degrees(rotation)
        rotation = axBig.transData.transform_angles(np.array((rotation,)),
                                                    l2.reshape((1, 2)))[0]
        text = ''
        if all_z_fit:
            text = r'${\rm All}~z~{\rm Fit~FMR}$'
        else:
            text =  r'$z=0~{\rm Fit~FMR}$'
        axBig.text( l2[0],l2[1], text, rotation = rotation, rotation_mode='anchor', fontsize=40 )

    plt.tight_layout()   
        
    save_str = "Figure4" if all_z_fit else "Figure1"
    save_str += ("_" + sim.lower()) if sim != "EAGLE" else ""
    save_str += ".pdf"
    
    plt.savefig( savedir + save_str, bbox_inches='tight' )
    plt.clf()

def make_MZR_prediction_fig(sim,all_z_fit,ax_real,ax_fake,ax_offsets,
                            STARS_OR_GAS="gas",savedir="./",
                            function = linear,
                            THRESHOLD = -5.00E-01,
                            width = 0.05, step = 0.1,
                            min_samp = 15):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,all_z_fit,THRESHOLD=THRESHOLD)

    min_alpha, *params = get_z0_alpha(sim, function=function)
    best_mu = star_mass - min_alpha*np.log10(SFR)
    if (all_z_fit):
        params, cov = curve_fit(function, best_mu, Z_use)
        
    plot_mu = np.linspace(np.min(best_mu),np.max(best_mu),100)
    best_line = function(plot_mu, *params)
    
    unique = np.unique(redshifts)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    sum_residuals = 0
    n_resids = 0
    MSE = 0
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width,step=step,
                                                       min_samp=min_samp)
        
        color = colors[index]
        
        mu = MZR_M_real - min_alpha * np.log10(real_SFR)
        
        lw = 2.5
        
        ax_real.plot( MZR_M_real, MZR_Z_real, color=color,
                      label=r'$z=%s$' %index, lw=lw )
        
        mu = MZR_M_real - min_alpha * np.log10(real_SFR)
        MZR_Z_fake = function(mu, *params)
        
        ax_fake.plot( MZR_M_real, MZR_Z_fake, color=color,
                      label=r'$z=%s$' %index, lw=lw )
        
        offset = MZR_Z_real - MZR_Z_fake

        sum_residuals += sum(offset**2)
        n_resids += len(offset)
        
        ax_offsets.plot( MZR_M_real, offset, color=color,
                         label=r'$z=%s$' %index, lw=lw )
        
        if index == len(snapshots) - 1:
            MSE = sum_residuals / n_resids
            print(f'\tMSE: {MSE:.3f} (dex)^2')
            txt_loc_x = 0.95
            ha = 'right'
            if all_z_fit:
                txt_loc_x = 0.05
                ha = 'left'
            ax_offsets.text( txt_loc_x, 0.07,
                             r'${\rm MSE} = \;$' + fr"${MSE:0.3f}$" + r'$\;({\rm dex })^2$',
                             transform=ax_offsets.transAxes, fontsize=16, ha=ha )
        
    return colors, MSE

if __name__ == "__main__":
    
    print("Hello World!")
