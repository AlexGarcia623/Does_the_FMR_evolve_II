import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

# from alpha_types import return_fake_MZR
from helpers import WHICH_SIM_TEX, plot_fake_MZR_find_alpha

savedir='./fake_MZR/'

sims = ['original','tng','eagle']
lCol = ['C1','C2','C0']
markers = ['^','*','o']
offsets = [0, 0.05, -0.05]

redshifts = np.arange(1,9)

for index,sim in enumerate(sims):
    plt.clf()
    fig, axs = plt.subplots(1,3,figsize=(12,5))
    
    ax_real = axs[0]
    ax_fake = axs[1]
    ax_offsets = axs[2]
    
    plot_fake_MZR_find_alpha(sim,ax_real,ax_fake,ax_offsets)

    for ax in axs:
        ax.set_xlabel(r'$\log M_*$')
    
    ymin = min(ax_fake.get_ylim()[0],ax_real.get_ylim()[0])
    ymax = max(ax_fake.get_ylim()[1],ax_real.get_ylim()[1])
    
    for ax in axs[:-1]:
        ax.set_ylim(ymin, ymax)
    
    ax_real.set_ylabel(r'$Z_{\rm true}$')
    ax_fake.set_ylabel(r'$Z_{\rm pred}$')
    ax_offsets.set_ylabel(r'$Z_{\rm true} - Z_{\rm pred}$')
    
    ax_fake.text(0.5,1.05,WHICH_SIM_TEX[sim.upper()],transform=ax_fake.transAxes,
                 ha='center')
    
    ax_real.text(0.5,0.9,r'${\rm MZR}_{\rm true}$',transform=ax_real.transAxes,
                 ha='center')
    ax_fake.text(0.5,0.9,r'${\rm MZR}_{{\rm pred}}$',
                 transform=ax_fake.transAxes, ha='center')
    
    ax_offsets.axhline(0.0, color='k', linestyle=':')
    
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.30)
    
    plt.savefig( savedir + '%s_MZR_adjusted.pdf' %sim, bbox_inches='tight')