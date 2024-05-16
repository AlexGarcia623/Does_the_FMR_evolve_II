import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from plotting import (
    make_MZR_fig, observational_MZR_data,
    get_n_movie_frames
)
from helpers import WHICH_SIM_TEX

mpl.rcParams['font.size'] = 16

sims = ['original','tng','eagle','simba']

savedir = './movie/'
STARS_OR_GAS = "gas".upper()

n_frames = get_n_movie_frames("ORIGINAL")

for frame in range(n_frames):
    plt.clf()
    print(f"Frame:", str(frame).zfill(3))
    fig, axs_MZR = plt.subplots(2, 2, figsize = (8,5.75), sharex=True, sharey=True)

    axs = axs_MZR.flatten()

    for index, ax in enumerate(axs):
        sim = sims[index].upper()

        bin_index = frame
        no_line = False
        
        if frame < 9 and sim == "SIMBA":
            no_line = True
        elif sim == "SIMBA":
            bin_index = frame - 9
        
        colors = make_MZR_fig(sim, ax, bin_index=bin_index, no_line=no_line)

        ax.text( 0.05, 0.825, WHICH_SIM_TEX[sim],
                 transform=ax.transAxes )

    ymin, ymax = axs[0].get_ylim()
    axs[0].set_ylim(6.6, ymax*1.03)
    xmin, _ = axs[0].get_xlim()
    axs[0].set_xlim(xmin, 12.1)
    axs[0].set_xticks([8,9,10,11])

    leg = axs[1].legend(frameon=True,labelspacing=0.05,
                        handletextpad=0, handlelength=0, 
                        markerscale=-1,bbox_to_anchor=(1,1.05),
                        framealpha=1, edgecolor='black',fancybox=False)
    for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
    for index, text in enumerate(leg.get_texts()):
        text.set_color(colors[index])
    leg.get_frame().set_linewidth(2.5)
    # leg.get_frame().set_facecolor('lightgray')

    # for ax in axs:
    #     observational_MZR_data(ax)

    axs[0].text(-0.1,0,r'$\log ({\rm O/H}) + 12 ~({\rm dex})$',
                ha='center', va='center', rotation=90, transform=axs[0].transAxes)
    axs[2].text(1.0,-0.175,r'$\log (M_*~[M_\odot])$', ha='center',
                transform=axs[2].transAxes)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    plt.savefig(savedir + '%s.png' %(str(frame).zfill(3)), bbox_inches='tight', dpi=300)
    plt.clf()