import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmasher as cmr

from helpers import WHICH_SIM_TEX, get_z0_alpha, get_all_redshifts

def make_FMR_fig(sim,all_z_fit,STARS_OR_GAS="gas",savedir="./"):
    
    STARS_OR_GAS = STARS_OR_GAS.upper()
    sim = sim.upper()
    
    star_mass, SFR, Z_use, redshifts = get_all_redshifts(sim,all_z_fit)

    min_alpha, _a_, _b_ = get_z0_alpha(sim)
    best_mu = star_mass - min_alpha*np.log10(SFR)
    if (all_z_fit):
        _a_, _b_ = np.polyfit(best_mu, Z_use, 1)
    best_line = _a_ * best_mu + _b_

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

    mappable = axBig.pcolormesh( xedges, yedges, hist, vmin = 0, vmax = 8, cmap=CMAP_TO_USE, rasterized=True )

    cbar = plt.colorbar( mappable, label=r"${\rm Redshift}$", orientation='horizontal' )
    cbar.ax.set_xticks(np.arange(0,9))
    cbar.ax.set_xticklabels(np.arange(0,9))

    axBig.plot( best_mu, best_line, color='k', lw=6.0 )

    if (STARS_OR_GAS == "GAS"):
        plt.ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$')
    elif (STARS_OR_GAS == "STARS"):
        plt.ylabel(r'$\log(Z_* [Z_\odot])$')
    plt.xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))

    if all_z_fit:
        axBig.text( 0.75, 0.1, r"${\rm All}~z~{\rm fit}$", transform=axBig.transAxes, ha='center' )
    # axBig.text( 0.75, 0.1 , r"${\rm Local~ FMR}$"  , transform=axBig.transAxes, ha='center' )

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

        ax.pcolormesh( current_x, current_y, current_hist, vmin = vmin, vmax = vmax, cmap=CMAP_TO_USE, rasterized=True )

        ax.plot( best_mu, best_line, color='k', lw=4.5 )

        ax.text( 0.65, 0.1, r"$z = %s$" %int(time), transform=plt.gca().transAxes )

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    x=9.75
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
    axBig.text( l2[0],l2[1], text, rotation = rotation, rotation_mode='anchor', fontsize=36 )

    plt.tight_layout()
        
        
    figure_name = name_figure(sim, all_z_fit)
    plt.savefig( savedir + '%s' %figure_name, bbox_inches='tight' )
    plt.clf()


def name_figure(sim, all_z_fit):
    name = 'Figure'
    pdf = '.pdf'
    
    appendix = ''
    
    if sim.upper() != 'EAGLE':
        appendix = "A"
        fig_number = 1
        if sim.upper() == "TNG":
            fig_number += 2
        if all_z_fit:
            fig_number += 1
    else:
        fig_number = 1
        if all_z_fit:
            fig_number = 3
    
    return name + appendix + str(fig_number) + pdf