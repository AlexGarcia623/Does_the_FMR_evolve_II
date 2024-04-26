import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.interpolate import interp1d

from alpha_types import (
    get_delta_evo, get_avg_relations, median
)
from helpers import (
    WHICH_SIM_TEX, get_z0_alpha
)
from plotting import (
    linear
)

sims = ['EAGLE','ORIGINAL','TNG']

redshifts = np.arange(0,9)

cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(redshifts))
newcolors = np.linspace(0, 1, len(redshifts))
colors = [ cmap(x) for x in newcolors[::-1] ]

for sim in sims:
    fig, axs = plt.subplots(1,4,figsize=(11,4),
                            gridspec_kw={'width_ratios': [1, 1, 0.45, 1]})

    delta_evo, lower, upper = get_delta_evo(sim)

    z0_SFMS, z0_MZR = None, None

    sum_residuals = 0
    n_data = 0

    min_alpha, *_ = get_z0_alpha(sim,function=linear)
    
    for index, redshift in enumerate(redshifts):
        mbins, zbins, Sbins = get_avg_relations(sim,redshift)

        if index == 0:
            z0_mbins, z0_zbins, z0_Sbins = mbins, zbins, Sbins
            z0_mu = z0_mbins - min_alpha * np.log10(z0_Sbins)
            params = np.polyfit(z0_mu, z0_zbins, 1)
            
            z0_zpred = linear(z0_mu, *params)
            
            z0_SFMS = interp1d(mbins, Sbins, fill_value='extrapolate')
            z0_MZR = interp1d(z0_mbins, z0_zpred, fill_value='extrapolate')

        axs[0].plot(mbins, zbins, color=colors[index],
                    label="$z=%s$" %index, lw=2)

        Z_pred = -delta_evo[index] * np.log10( Sbins / z0_SFMS(mbins) ) + z0_MZR(mbins)

        axs[1].plot(mbins, Z_pred, color=colors[index],
                    label="$z=%s$" %index, lw=2)

        residuals = zbins - Z_pred
        sum_residuals += sum(residuals**2)
        n_data += len(residuals)

        axs[3].plot(mbins, residuals, color=colors[index],
                    label="$z=%s$" %index, lw=2)

    MSE = sum_residuals / n_data
    axs[3].text(0.95, 0.07,
                r'${\rm MSE} = \;$' + fr"${MSE:0.3f}$" + r'$\;({\rm dex })^2$',
                transform=axs[3].transAxes, fontsize=16, ha='right')

    ax_real = axs[0]
    ax_fake = axs[1]
    ax_blank = axs[2]
    ax_offsets = axs[3]

    ax_blank.axis('off')

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

    ax_fake.text(0.95,0.07,WHICH_SIM_TEX[sim.upper()],transform=ax_fake.transAxes,
                 ha='right')

    ax_real.text(0.05,0.875,r'${\rm True}$',transform=ax_real.transAxes)
    ax_fake.text(0.05,0.875,r'${\rm FMR~Prediction}$',transform=ax_fake.transAxes)

    ax_offsets.axhline(0.0, color='k', linestyle=':')

    leg = ax_offsets.legend(frameon=False,labelspacing=0.05,
                            handletextpad=0, handlelength=0, 
                            markerscale=-1,bbox_to_anchor=(1,1))
    for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)
    for index, text in enumerate(leg.get_texts()):
        text.set_color(colors[index])

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.0)

    plt.savefig('./Figures (pdfs)/' + f'{sim}_phenom' + '.pdf', bbox_inches='tight')