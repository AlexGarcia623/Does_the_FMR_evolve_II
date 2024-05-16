import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import cmasher as cmr
import sys

sys.path.insert(1,'Data/')

from helpers import (
    WHICH_SIM_TEX, get_z0_alpha, get_all_redshifts,
    get_one_redshift, get_medians, switch_sim, get_allz_alpha,
    get_idv_alpha
)
from alpha_types import (
    get_avg_relations
)
from additional_data import (
    Tremonti_04_x, Tremonti_04_y,
    Zahid_11_x, Zahid_11_y,
    Sanders_2021_x_z2, Sanders_2021_y_z2,
    Sanders_2021_x_z3, Sanders_2021_y_z3,
    Heintz_2023_x, Heintz_2023_y,
    Nakajima_2023_MZR_x, Nakajima_2023_MZR_y
)

def format_func(value, tick_number):
    return "{:.1f}".format(value)

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
    print(min_alpha)
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
                            width = 0.1, step = 0.1,
                            min_samp = 20):
    
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
    
    z0_MZR = None
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width,
                                                       min_samp=min_samp)
        if index == 0:
            z0_MZR = interp1d(MZR_M_real, MZR_Z_real, fill_value='extrapolate')
        
        color = colors[index]
        
        mu = MZR_M_real - min_alpha * np.log10(real_SFR)
        
        lw = 3
        
        ax_real.plot( MZR_M_real, MZR_Z_real, color=color,
                      label=r'$z=%s$' %index, lw=lw )
        
        mu = MZR_M_real - min_alpha * np.log10(real_SFR)
        MZR_Z_fake = function(mu, *params)
        
        ax_fake.plot( MZR_M_real, MZR_Z_fake, color=color,
                      label=r'$z=%s$' %index, lw=lw )
        
        offset = MZR_Z_real - MZR_Z_fake
        # true_offset = z0_MZR(MZR_M_real) - MZR_Z_real
        # offset = pred_offset/true_offset
        
        sum_residuals += sum(offset**2)
        n_resids += len(offset)
        
        ax_offsets.plot( MZR_M_real, offset, color=color,
                         label=r'$z=%s$' %index, lw=lw )
        
        if index == len(snapshots) - 1:
            MSE = sum_residuals / n_resids
            print(f'\tMSE: {MSE:.3f} (dex)^2')
            txt_loc_x = 0.075
            ha = 'left'
            # if sim != "":
            #     txt_loc_x = 0.95
            #     ha = 'right'
            ax_offsets.text( txt_loc_x, 0.07,
                             r'${\rm MSE} = \;$' + fr"${MSE:0.3f}$" + r'$\;({\rm dex })^2$',
                             transform=ax_offsets.transAxes, fontsize=16, ha=ha )
        
    return colors, MSE

def get_n_movie_frames(sim,STARS_OR_GAS="gas"):
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    snap = snapshots[0]
    
    star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
    width = 0.1
    min_samp = 20
    MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                   width=width,
                                                   min_samp=min_samp)

    return len(MZR_M_real)
    
def make_MZR_fig(sim,ax,STARS_OR_GAS="gas",
                 THRESHOLD = -5.00E-01,
                 width = 0.1, step = 0.1,
                 min_samp = 20, bin_index=0, 
                 no_line=False):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_z_fit = False
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,all_z_fit,THRESHOLD=THRESHOLD)
    
    unique = np.unique(redshifts)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    Z_min_bin = []
    m_min = 0
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width,
                                                       min_samp=min_samp)
        
        MZR_M_real -= width/2
        
        if bin_index > len(MZR_Z_real) - 1:
            Z_min_bin.append(np.nan)
        elif (len(MZR_Z_real) > 0):
            Z_min_bin.append(MZR_Z_real[bin_index])
        else:
            Z_min_bin.append(np.nan)
        
        if index == 0:
            m_min = MZR_M_real[bin_index]
        
        color = colors[index]
        
        lw = 2.5
        
        ax.plot( MZR_M_real, MZR_Z_real, color=color,
                      label=r'$z=%s$' %index, lw=lw )
    
    Z_min_bin_no_subtract = np.array(Z_min_bin)
    Z_min_bin -= Z_min_bin_no_subtract[0]

    xstart = 10.25
    ystart = 7.05
    xlen = 1.6
    ylen = 1
        
    ax_inset = ax.inset_axes([xstart, ystart, xlen, ylen], transform=ax.transData)
    
    if not no_line:
        ax_inset.scatter(unique, Z_min_bin, c=colors, s=10)
        
    ax_inset.spines['bottom'].set_linewidth(1); ax_inset.spines['top'].set_linewidth(1)
    ax_inset.spines['left'].set_linewidth(1)  ; ax_inset.spines['right'].set_linewidth(1)
        
    ax_inset.tick_params(axis='both', which='major', length=3, width=1.5)
    ax_inset.tick_params(axis='both', which='minor', length=2, width=1)
    
    ax_inset.tick_params(axis='y', labelsize=10)
    ax_inset.tick_params(axis='x', which='both', top=True, labelsize=10)
    
    ax_inset.set_ylim(-1.1,0.2)
    ax_inset.set_yticks([-1,-0.5,0])
    ax_inset.minorticks_on()
    ax_inset.set_xlim(-1,9)
    
    _y_ = 0.5
    if sim == "SIMBA":
        _y_ = 0.4
    ax_inset.text(-0.4,_y_,r'${\rm Offset~(dex)}$', fontsize=10,
                  transform=ax_inset.transAxes, va='center', rotation=90)
    ax_inset.text(0.5,-0.3,r'${\rm Redshift}$', fontsize=10,
                  transform=ax_inset.transAxes, ha='center')
    
    ax_inset.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax_inset.set_xticks([0,4,8])
    
    ymin = np.nanmin(Z_min_bin_no_subtract)
    ymax = np.nanmax(Z_min_bin_no_subtract)
    
    x = np.linspace(m_min, xstart, 100)
    
    slope1 = (ystart - ymin) / (xstart - m_min)
    slope2 = (ystart + ylen - ymax) / (xstart - m_min)
    
    y1 = slope1 * (x - xstart) + ystart
    y2 = slope2 * (x - xstart) + ystart + ylen
    
    if not no_line:
        ax.plot(x, y1, color='k', alpha=0.5, lw=1)
        ax.plot(x, y2, color='k', alpha=0.5, lw=1)
        ax.vlines(x=m_min, ymin=ymin, ymax=ymax, color='k', lw=1)
    
    return colors

def make_SFMS_fig(sim,ax,STARS_OR_GAS="gas",
                  THRESHOLD = -5.00E-01,
                  width = 0.1, step = 0.1,
                  min_samp = 20):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_z_fit = False
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,all_z_fit,THRESHOLD=THRESHOLD)
    
    unique = np.unique(redshifts)
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    
    S_min_bin = []
    m_min = 0
    
    for index, snap in enumerate(snapshots):
        
        star_mass, Z_true, SFR = get_one_redshift(BLUE_DIR,snap,
                                                  STARS_OR_GAS=STARS_OR_GAS)
        
        MZR_M_real, MZR_Z_real, real_SFR = get_medians(star_mass,Z_true,SFR,
                                                       width=width, min_samp=min_samp)
        
        MZR_M_real -= width/2
        
        if len(real_SFR) > 0:
            S_min_bin.append(np.log10(real_SFR[0]))
        else:
            S_min_bin.append(np.nan)
        
        if index == 0:
            m_min = MZR_M_real[0]
            
            a,b = np.polyfit(MZR_M_real, np.log10(real_SFR), 1)
            
            print(f"SFMS Slope @ z=0: {a:0.3f}")
        
        color = colors[index]
        
        lw = 2.5
        
        ax.plot( MZR_M_real, np.log10(real_SFR), color=color,
                      label=r'$z=%s$' %index, lw=lw )

    S_min_bin_no_subtract = np.array(S_min_bin)
    S_min_bin -= S_min_bin_no_subtract[0]
        
    xstart = 9.6
    ystart = -3
    xlen = 1.8
    ylen = 2.2
        
    ax_inset = ax.inset_axes([xstart, ystart, xlen, ylen], transform=ax.transData)
    ax_inset.scatter(unique, S_min_bin, c=colors, s=10)
        
    ax_inset.spines['bottom'].set_linewidth(1); ax_inset.spines['top'].set_linewidth(1)
    ax_inset.spines['left'].set_linewidth(1)  ; ax_inset.spines['right'].set_linewidth(1)
        
    ax_inset.tick_params(axis='both', which='major', length=3, width=1.5)
    ax_inset.tick_params(axis='both', which='minor', length=2, width=1)
    
    ax_inset.tick_params(axis='y', labelsize=10)
    ax_inset.tick_params(axis='x', which='both', top=True, labelsize=10)
    
    ax_inset.set_ylim(-0.25,2.25)
    ax_inset.set_xlim(-1,9)
    
    ax_inset.text(-0.2,0.5,r'${\rm Offset~(dex)}$', fontsize=10,
                  transform=ax_inset.transAxes, va='center', rotation=90)
    ax_inset.text(0.5,-0.35,r'${\rm Redshift}$', fontsize=10,
                  transform=ax_inset.transAxes, ha='center')
    
    ax_inset.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax_inset.set_xticks([0,4,8])
    ax_inset.set_yticks([0,1,2])
    
    ymin = np.nanmin(S_min_bin_no_subtract)
    ymax = np.nanmax(S_min_bin_no_subtract)
    
    x = np.linspace(m_min, xstart, 100)
    
    slope1 = (ystart - ymin) / (xstart - m_min)
    slope2 = (ystart + ylen - ymax) / (xstart - m_min)
    
    y1 = slope1 * (x - xstart) + ystart
    y2 = slope2 * (x - xstart) + ystart + ylen
    
    ax.plot(x, y1, color='k', alpha=0.5, lw=1)
    ax.plot(x, y2, color='k', alpha=0.5, lw=1)
    ax.vlines(x=m_min, ymin=ymin, ymax=ymax, color='k', lw=1)
        
    return colors

def observational_MZR_data(ax):
    '''Abandoned'''
    ax.plot(Tremonti_04_x, Tremonti_04_y, color='k', alpha=0.5)
    
    ax.plot(Zahid_11_x, Zahid_11_y, color='k', alpha=0.5, linestyle=':')
    
    ax.plot(Sanders_2021_x_z2, Sanders_2021_y_z2, color='k', alpha=0.5, linestyle='--')
    
    ax.plot(Sanders_2021_x_z3, Sanders_2021_y_z3, color='k', alpha=0.5, linestyle='-.')
    
    ax.plot(Heintz_2023_x, Heintz_2023_y, color='k', alpha=0.5, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    
    ax.plot(Nakajima_2023_MZR_x, Nakajima_2023_MZR_y, color='k', alpha=0.5, linestyle=(0, (3, 5, 1, 5, 1, 5)))

def make_alpha_evo_plot(sim, redshifts=[2]):
    sim = sim.upper()
    z0_mbins, z0_zbins, z0_Sbins = None, None, None
    
    assert(len(redshifts) == 1) ## Deprecated functionality
    
    fig = plt.figure(figsize=(7,6))
    
    alpha_evo = np.zeros(len(redshifts))
    evo_min   = np.zeros(len(redshifts))
    evo_max   = np.zeros(len(redshifts))
    
    z0_mbins, z0_zbins, z0_Sbins = get_avg_relations(sim, 0)
    
    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=9)
    newcolors = np.linspace(0, 1, 9)
    colors = [ cmap(x) for x in newcolors[::-1] ]
        
    for index, redshift in enumerate(redshifts):
        ax = plt.gca()
        
        mbins, zbins, Sbins = get_avg_relations(sim, redshift)
            
        alphas = np.linspace(0,1,100)
        disp   = np.zeros(len(alphas))

        all_mbins = np.append(mbins,z0_mbins)
        all_zbins = np.append(zbins,z0_zbins)
        all_Sbins = np.append(Sbins,z0_Sbins)
        all_param = []
        
        for j, alpha in enumerate(alphas):
            mu = all_mbins - alpha * np.log10(all_Sbins)
            
            params = np.polyfit(mu, all_zbins, 1)
            all_param.append(params)
            interp_line = np.polyval(params, mu)
            
            disp[j] = np.std( np.abs(all_zbins) - np.abs(interp_line) )
            
        argmin = np.argmin(disp)
        min_alpha = alphas[argmin]
        min_disp  = disp[argmin]
        best_param = all_param[argmin]
        
        width = 1.05 * min_disp
        
        within_uncertainty = alphas[ (disp < width) ]

        min_uncertain = within_uncertainty[0]
        max_uncertain = within_uncertainty[-1] 
        
        alpha_evo[index] = min_alpha
        evo_min[index] = min_uncertain
        evo_max[index] = max_uncertain
        
        s=45
        ax.scatter(mbins - min_alpha * np.log10(Sbins), zbins, color=colors[redshift],s=s)
        ax.scatter(z0_mbins - min_alpha * np.log10(z0_Sbins), z0_zbins, color=colors[0], marker='s',s=s)
        
        x_vals = all_mbins - min_alpha * np.log10(all_Sbins)
        _x_ = np.linspace(np.min(x_vals),np.max(x_vals), 100)
        _y_ = best_param[0] * _x_ + best_param[1]
        
        ax.plot(_x_, _y_, color='k',lw=3)
        
        xlabel = (r"$\langle\mu_{%0.2f}\rangle" %min_alpha + 
                  fr" = \langle \log M_*\rangle - {min_alpha:0.2f}\langle\log" 
                  + r"{\rm SFR}\rangle$")
        
        ax.set_xlabel(xlabel)
        
        ax.text(0.05,0.8,r'$z=0$',color=colors[0],transform=ax.transAxes)
        ax.text(0.05,0.725,r'$z=%s$' %redshift, color=colors[redshift], transform=ax.transAxes)
        
        ax.text(0.933,0.575,r'$\alpha_{{\rm evo},~z=0\to%s} = %0.2f_{%0.2f}^{%0.2f}$'%(
                    redshift,min_alpha,min_uncertain,max_uncertain
                ),transform=ax.transAxes,ha='right',fontsize=18)
        
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, 11.25)
        
        xstart = 9.75
        ystart = 7.975
        xlen = 1.4
        ylen = 0.45

        ax_bot = ax.inset_axes([xstart, ystart, xlen, ylen], transform=ax.transData)
        
        ax_bot.plot(alphas, disp, color='blue', lw=2)
        ax_bot.axvline(min_alpha, color='k', linestyle='--')
        ax_bot.axvline(min_uncertain, color='gray', linestyle=':')
        ax_bot.axvline(max_uncertain, color='gray', linestyle=':')
        
        ax_bot.spines['bottom'].set_linewidth(1); ax_bot.spines['top'].set_linewidth(1)
        ax_bot.spines['left'].set_linewidth(1)  ; ax_bot.spines['right'].set_linewidth(1)

        ax_bot.tick_params(axis='both', which='major', length=5, width=1.5)
        ax_bot.tick_params(axis='both', which='minor', length=3, width=1)
        
        ax_bot.tick_params(axis='y', labelsize=15)
        ax_bot.tick_params(axis='x', which='both', top=True, labelsize=15)
        
        xmin, xmax = ax_bot.get_xlim()
        ax_bot.set_xlim(-0.1,1.1)
        
        ymin, ymax = ax_bot.get_ylim()
        ax_bot.set_ylim(ymin*0.8, ymax)
        
        ax_bot.set_ylabel(r'${\rm Dispersion}$',fontsize=15)
        ax_bot.set_xlabel(r'$\alpha$',fontsize=15)
    
    ax.set_ylabel(r'$\log ({\rm O/H}) + 12~({\rm dex})$')
    
    ax.text(0.05,0.875,WHICH_SIM_TEX[sim],color='k',transform=ax.transAxes)
    
    plt.tight_layout()
    
    plt.subplots_adjust(wspace=0,hspace=0.4)
    plt.savefig('./Figures (pdfs)/' + 'alpha_demo.pdf', bbox_inches='tight')
    
    return 

if __name__ == "__main__":
    
    print("Hello World!")
