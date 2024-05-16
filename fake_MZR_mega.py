import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plotting import (
    make_MZR_prediction_fig,
    linear, fourth_order, format_func
)    
from helpers import WHICH_SIM_TEX

mpl.rcParams['axes.linewidth'] = 3.5
mpl.rcParams['xtick.major.width'] = 2.75
mpl.rcParams['ytick.major.width'] = 2.75
mpl.rcParams['xtick.minor.width'] = 1.75
mpl.rcParams['ytick.minor.width'] = 1.75
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['ytick.minor.size'] = 5

sims = ['original','tng','eagle','simba']

savedir = './Figures (pdfs)/'
STARS_OR_GAS = "gas".upper()

fig, axs_all = plt.subplots(4,4,figsize=(11,13),
                            gridspec_kw={'width_ratios': [1, 1, 0.4, 1]},
                            sharex=True)

all_z_fit = False
function = linear

ax_column1 = []
ax_column2 = []
ax_column3 = []

YMIN, YMAX = 0,0

for index, sim in enumerate(sims):
    axs = axs_all[index,:]
    ax_real = axs[0]
    ax_column1.append(ax_real)
    ax_fake = axs[1]
    ax_column2.append(ax_fake)
    ax_blank = axs[2]
    ax_offsets = axs[3]
    ax_column3.append(ax_offsets)

    ax_blank.axis('off')

    colors, MSE = make_MZR_prediction_fig(sim,all_z_fit, ax_real, ax_fake,
                                          ax_offsets,function = function)

    if index == 3:
        for ax in axs:
            ax.set_xlabel(r'$\log(M_*~[M_\odot])$')
    # if index == 0:
    #     ax_fake.text(0.05,0.9,r'$z=0~{\rm Fit~FMR}$', transform=ax_fake.transAxes, fontsize=14)
    # else:
    for ax in axs:
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    ymin = min(ax_real.get_ylim()[0], ax_fake.get_ylim()[0])
    ymax = max(ax_real.get_ylim()[1], ax_fake.get_ylim()[1])

    for ax in [ax_real, ax_fake]:
        ax.set_ylim(ymin, ymax)

    # ax_fake.sharex(ax_real)
    # ax_offsets.sharex(ax_real)
    ax_real.set_xticks([8,9,10,11])

    ax_real.set_ylabel(r'$\log({\rm O/H}) + 12~{\rm (dex)}$')
    ax_fake.set_yticklabels([])
    ax_offsets.set_ylabel(r'${\rm True} - {\rm Predicted}$')

    ax_real.text(0.05,0.85,WHICH_SIM_TEX[sim.upper()],transform=ax_real.transAxes)

    if index == 0:
        ax_real.text(0.5,1.05,r'${\rm True~MZR}$',transform=ax_real.transAxes,ha='center',fontsize=28)
        ax_fake.text(0.5,1.05,r'${\rm Predicted~MZR}$',transform=ax_fake.transAxes,ha='center',fontsize=28)
        ax_offsets.text(0.5,1.05,r'${\rm Residuals}$',transform=ax_offsets.transAxes,ha='center',fontsize=28)

    ax_offsets.axhline(0.0, color='k', linestyle=':', lw=3)

    if index == 0:
        YMIN, YMAX = ax_real.get_ylim()
    
    ax_fake.set_ylim(YMIN, YMAX)
    ax_real.set_ylim(YMIN, YMAX)

    ax_offsets.set_ylim(-0.7,0.15)
    
    if index == 0:
        leg = ax_offsets.legend(frameon=True,labelspacing=0.05,
                                handletextpad=0, handlelength=0, 
                                markerscale=-1,bbox_to_anchor=(1,1.05),
                                framealpha=1, edgecolor='k',fancybox=False)
        for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
        for index, text in enumerate(leg.get_texts()):
            text.set_color(colors[index])
        leg.get_frame().set_linewidth(3)

# ymin = np.min([
#     [ax.get_ylim()[0] for ax in ax_column1],
#     [ax.get_ylim()[0] for ax in ax_column2]
# ])
# ymax = np.max([
#     [ax.get_ylim()[1] for ax in ax_column1],
#     [ax.get_ylim()[1] for ax in ax_column2]
# ])

# for ax in ax_column1:
#     ax.set_ylim(ymin,ymax)
# for ax in ax_column2:
#     ax.set_ylim(ymin,ymax)
        
plt.tight_layout()

plt.subplots_adjust(wspace=0.0, hspace=0.0)

plt.savefig( savedir + 'fake_MZR_mega.pdf', bbox_inches='tight')