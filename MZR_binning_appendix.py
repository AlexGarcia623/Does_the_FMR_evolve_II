import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmasher as cmr

from plotting import (
    make_MZR_prediction_fig, linear
)

savedir = './Figures (pdfs)/'
STARS_OR_GAS = "gas".upper()

sims = ['original','tng','eagle']
sim_vals = [0, 1, 2]

offset = [-0.05,-0.025,0.025,0.05]

fig, axs = plt.subplots(3,1,figsize=(6,12),sharey=True,sharex=True)
markersize = 60

### Bin Width First ###
bin_width = [0.025,0.05,0.1,0.5]
bin_step = [i*2 for i in bin_width]

cmap = cmr.lavender
num_colors = len(bin_width)
intervals = np.linspace(0.15, 0.85, num_colors)
colors = [cmap(interval) for interval in intervals]

labels = [r'$0.025~{\rm dex}$',
          r'$0.05~{\rm dex~(fiducial)}$',
          r'$0.1~{\rm dex}$',
          r'$0.5~{\rm dex}$']

for i, all_z_fit in enumerate([False, True]):
    marker = 'v' if all_z_fit else '^'
    for j, sim in enumerate(sims):
        for k, width in enumerate(bin_width):
            step = bin_step[k]
            dummy_fig, dummy_axs = plt.subplots(1,3)

            _, MSE = make_MZR_prediction_fig(sim,all_z_fit,dummy_axs[0],
                                             dummy_axs[1], dummy_axs[2],
                                             function = linear, width=width,
                                             step = step)
            if j == 0 and i == 0:
                ## only do this once
                axs[0].scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                                label=labels[k], s=markersize )
            else:
                axs[0].scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                                s=markersize)
            plt.close(dummy_fig)
            
leg = axs[0].legend(frameon=False, loc='best',
                    handletextpad=0, handlelength=0, 
                    markerscale=-1)
for n, text in enumerate( leg.texts ):
    text.set_color( colors[n] )
for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)

axs[0].text( 0.95, 0.95, r'${\rm MZR~Bin~Width}$', ha='right', va='top', transform=axs[0].transAxes )

### Bin Step Next ###
bin_step = [0.05,0.1,0.5,1.0]
bin_width = [i/2 for i in bin_step]

cmap = cmr.ember
num_colors = len(bin_width)
intervals = np.linspace(0.15, 0.85, num_colors)
colors = [cmap(interval) for interval in intervals]

labels = [r'$0.05~{\rm dex}$',
          r'$0.1~{\rm dex~(fiducial)}$',
          r'$0.5~{\rm dex}$',
          r'$1.0~{\rm dex}$']

for i, all_z_fit in enumerate([False, True]):
    marker = 'v' if all_z_fit else '^'
    for j, sim in enumerate(sims):
        for k, width in enumerate(bin_width):
            step = bin_step[k]
            dummy_fig, dummy_axs = plt.subplots(1,3)

            _, MSE = make_MZR_prediction_fig(sim,all_z_fit,dummy_axs[0],
                                             dummy_axs[1], dummy_axs[2],
                                             function = linear, width=width,
                                             step = step)
            if j == 0 and i == 0:
                ## only do this once
                axs[1].scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                                label=labels[k], s=markersize )
            else:
                axs[1].scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                                s=markersize)
            plt.close(dummy_fig)
            
leg = axs[1].legend(frameon=False, loc='best',
                    handletextpad=0, handlelength=0, 
                    markerscale=-1)
for n, text in enumerate( leg.texts ):
    text.set_color( colors[n] )
for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)

axs[1].text( 0.95, 0.95, r'${\rm MZR~Bin~Separation}$', ha='right', va='top', transform=axs[1].transAxes )

### Number of Galaxies ###
n_gals = [10, 15, 25, 50]

cmap = cmr.cosmic
num_colors = len(n_gals)
intervals = np.linspace(0.15, 0.85, num_colors)
colors = [cmap(interval) for interval in intervals]

labels = [r'$10$',
          r'$15~{\rm(fiducial)}$',
          r'$50$',
          r'$100$']

for i, all_z_fit in enumerate([False, True]):
    marker = 'v' if all_z_fit else '^'
    for j, sim in enumerate(sims):
        for k, n_gal in enumerate(n_gals):
            dummy_fig, dummy_axs = plt.subplots(1,3)

            _, MSE = make_MZR_prediction_fig(sim,all_z_fit,dummy_axs[0],
                                             dummy_axs[1], dummy_axs[2],
                                             function = linear, 
                                             min_samp = n_gal)
            if j == 0 and i == 0:
                ## only do this once
                axs[2].scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                                label=labels[k], s=markersize )
            else:
                axs[2].scatter( sim_vals[j] + offset[k], MSE, color=colors[k], marker=marker,
                                s=markersize)
            plt.close(dummy_fig)
            
leg = axs[2].legend(frameon=False, loc='best',
                    handletextpad=0, handlelength=0, 
                    markerscale=-1)
for n, text in enumerate( leg.texts ):
    text.set_color( colors[n] )
for i in range(len(leg.get_texts())): leg.legendHandles[i].set_visible(False)

axs[2].text( 0.95, 0.95, r'${\rm Min~ Galaxies~ in~ Bin}$', ha='right', va='top', transform=axs[2].transAxes )


axs[0].xaxis.set_minor_locator(ticker.NullLocator())
axs[0].set_xticks(sim_vals)        
axs[0].set_xticklabels([r'${\rm Illustris}$',r'${\rm TNG}$',r'${\rm EAGLE}$'])

axs[1].set_ylabel(r'${\rm MSE}$')

plt.tight_layout()

plt.subplots_adjust(hspace=0.0)

plt.savefig(savedir + 'AppendixB.pdf', bbox_inches='tight')