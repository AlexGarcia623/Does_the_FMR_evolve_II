import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from plotting import (
    make_MZR_prediction_fig,
    linear, fourth_order
)
from helpers import WHICH_SIM_TEX

sims = ['original','tng','eagle']

savedir = './Figures (pdfs)/'
STARS_OR_GAS = "gas".upper()

for all_z_fit in [False, True]:
    for function in [linear]:#, fourth_order]:
        for sim in ['eagle']:#sims
            plt.clf()
            fig, axs = plt.subplots(1,4,figsize=(11,4),
                                    gridspec_kw={'width_ratios': [1, 1, 0.45, 1]})

            ax_real = axs[0]
            ax_fake = axs[1]
            ax_blank = axs[2]
            ax_offsets = axs[3]

            ax_blank.axis('off')
            
            colors = make_MZR_prediction_fig(sim,all_z_fit, ax_real, ax_fake,
                                             ax_offsets,function = function)


            for ax in axs:
                ax.set_xlabel(r'$\log M_*$')

            ymin = min(ax_real.get_ylim()[0], ax_fake.get_ylim()[0])
            ymax = max(ax_real.get_ylim()[1], ax_fake.get_ylim()[1])
            
            for ax in [ax_real, ax_fake]:
                ax.set_ylim(ymin, ymax)
                
            ax_fake.sharex(ax_real)
            ax_offsets.sharex(ax_real)
            ax_real.set_xticks([8,9,10,11])
            
            ax_real.set_ylabel(r'$\log({\rm O/H}) + 12~{\rm (dex)}$')
            ax_fake.set_yticklabels([])
            ax_offsets.set_ylabel(r'${\rm True} - {\rm Predicted}$')

            ax_fake.text(0.95,0.05,WHICH_SIM_TEX[sim.upper()],transform=ax_fake.transAxes,
                         ha='right')

            ax_real.text(0.05,0.875,r'${\rm True}$',transform=ax_real.transAxes)
            ax_fake.text(0.05,0.875,r'${\rm FMR~Prediction}$',transform=ax_fake.transAxes)
            if (all_z_fit):
                ax_fake.text(0.05,0.775,r'${\rm All~}z~{\rm fit}$',transform=ax_fake.transAxes)
            
            ax_offsets.axhline(0.0, color='k', linestyle=':')

            leg = ax_offsets.legend(frameon=False,labelspacing=0.05,
                                    handletextpad=0, handlelength=0, 
                                    markerscale=-1,bbox_to_anchor=(1,1))
            for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
            for index, text in enumerate(leg.get_texts()):
                text.set_color(colors[index])
            
            plt.tight_layout()

            plt.subplots_adjust(wspace=0.0)
            
            save_str = "Figure5" if all_z_fit else "Figure3"
            save_str += ".pdf"
            
            plt.savefig( savedir + save_str, bbox_inches='tight')