import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from alpha_types import (
    get_alpha_evo
)
from helpers import (
    ttest, estimate_symmetric_error, WHICH_SIM_TEX
)

savedir = './Figures (pdfs)'

mpl.rcParams['axes.linewidth'] = 3.5
mpl.rcParams['xtick.major.width'] = 2.75
mpl.rcParams['ytick.major.width'] = 2.75
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

sims = ['ORIGINAL','TNG','EAGLE','SIMBA']
cols = ['C1','C2','C0','C3']
mark = ['^','*','o','s']
ls   = [':','--','-.','-']
offs = [0.00,0.05,-0.05,-0.05]
ms   = 7

all_alpha = []
all_lower = []
all_upper = []

fig = plt.figure(figsize=(9,4))

z = np.arange(0,9)

for index, sim in enumerate(sims):
    alpha_evo, evo_min, evo_max = get_alpha_evo(sim)
    
    lower = alpha_evo - evo_min
    upper = evo_max - alpha_evo
    
    no_nans = ~np.isnan(alpha_evo[1:])
    measurements = (alpha_evo[1:])[no_nans]
    errors = estimate_symmetric_error( lower[1:][no_nans], upper[1:][no_nans] )
    
    weighted_mean = np.sum(measurements / errors**2) / np.sum(1 / errors**2)
    weighted_std_error = np.sqrt(1 / np.sum(1 / errors**2))
    
    all_alpha.append(alpha_evo)
    all_lower.append(lower)
    all_upper.append(upper)
    
    plt.errorbar( z[1:] + offs[index], alpha_evo[1:], yerr = [lower[1:],upper[1:]],
                  label=WHICH_SIM_TEX[sim], markersize=ms,
                  marker=mark[index], color=cols[index], linestyle='none', alpha=0.75 )

    plt.axhline(weighted_mean,alpha=0.5,color=cols[index],linestyle=ls[index])
    
leg  = plt.legend(frameon=True,handletextpad=0, handlelength=0,
                  markerscale=0,labelspacing=0.05)
lCol = ['C1','C2','C0','C3']
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )
    
# get handles
handles, labels = plt.gca().get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
# use them in the legend
leg = plt.legend(handles, labels, frameon=True,handletextpad=0.4, handlelength=0,labelspacing=0.01,
                 loc=(0.015, 0.05))
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )

leg.get_frame().set_alpha(1.0)
leg.get_frame().set_edgecolor('white')

plt.xlabel(r'${\rm Redshift}$')
plt.ylabel(r'$\alpha_{\rm evo}$')

xmin, xmax = plt.xlim()
plt.xlim(-0.825,xmax)
plt.xticks(np.arange(0,9,1))
plt.yticks([0,0.25,0.5,0.75,1])
plt.ylim(-0.025,1.025)
plt.tight_layout()

plt.savefig('Figures (pdfs)/'+"alpha_evo.pdf", bbox_inches='tight')


for index, alphas in enumerate(all_alpha):
    no_nans = ~np.isnan(alphas[1:]) ## Remove z=8 in SIMBA
    
    lower = (all_lower[index][1:])[no_nans]
    upper = (all_upper[index][1:])[no_nans]
    
    alphas = (alphas[1:])[no_nans]
    
    which_redshift_compare = 0
    hypothesized_value = alphas[which_redshift_compare]
    
    est_error = estimate_symmetric_error( lower, upper )
    
    print(f'\n{sims[index]}, compared to z={which_redshift_compare+1} alpha evo value')
    ttest(hypothesized_value, alphas, est_error)