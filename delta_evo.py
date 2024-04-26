import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from alpha_types import get_delta_evo, polynomial
from helpers import (
    ttest, estimate_symmetric_error, WHICH_SIM_TEX
)


savedir='./Figures (pdfs)/'

sims = ['original','tng','eagle']
lCol = ['C1','C2','C0']
markers = ['^','*','o']
offsets = [0, 0.05, -0.05]

alpha_scatt = [[0,0,0],[0,0,0],[0,0,0]]

redshifts = np.arange(0,9)

plt.clf()
fig = plt.figure(figsize=(9,4))

all_alpha = []
all_lower = []
all_upper = []

for index,sim in enumerate(sims):
    print(sim)
    alpha_evo, l, u = get_delta_evo(sim)
    
    lower = alpha_evo - l
    upper = u - alpha_evo
    
    all_alpha.append(alpha_evo)
    all_lower.append(lower)
    all_upper.append(upper)
    
    color = lCol[index]
    marker = markers[index]
    offset = offsets[index]
    ms = 7
    if marker == '*':
        ms = 10
    
    scatter = alpha_scatt[index]
    
    plt.errorbar( redshifts+offset, alpha_evo, label=WHICH_SIM_TEX[sim.upper()],
                  alpha=0.75, yerr = [lower, upper], color=color,
                  linestyle='none', marker=marker, markersize=ms)
    
    plt.errorbar(
        [offset], [scatter[0]],
        xerr=0,
        yerr=[[scatter[0] - scatter[2]], [scatter[1] - scatter[0]]],
        alpha=0.75, color=color, linestyle='none', marker=marker, markersize=ms,
        markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5
    )
    
plt.xlabel(r'${\rm Redshift}$')
plt.ylabel(r'$\delta_{\rm evo}$')

# plt.text(0.18,0.5,r'$\alpha_{\rm scatter}$',ha='center',va='center', fontsize=14, rotation=90)
# plt.axvline(0.3,color='k',linestyle='--')

# plt.ylim(-0.1,1.1)

leg  = plt.legend(frameon=False,handletextpad=0, handlelength=0,
                  markerscale=0,loc='upper right',labelspacing=0.05)
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )
# get handles
handles, labels = plt.gca().get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
# use them in the legend
leg = plt.legend(frameon=False,handletextpad=0.75, handlelength=0,labelspacing=0.01,
                 loc='upper left')
for n, text in enumerate( leg.texts ):
    text.set_color( lCol[n] )

plt.tight_layout()

plt.savefig( savedir + 'delta_evo.pdf',bbox_inches='tight')

### Do reference value t-test
    
for index,alphas in enumerate(all_alpha):
    lower = all_lower[index]
    upper = all_upper[index]
    
    alphas = alphas[1:]
    lower = lower[1:]
    upper = upper[1:]
    
    which_redshift_compare = 1 - 1 # subtract one, we removed z=0
    hypothesized_value = alphas[which_redshift_compare]
    
    est_error = estimate_symmetric_error( lower, upper )
    
    print(f'{sims[index]}, compared to z={which_redshift_compare} alpha value')
    ttest(hypothesized_value, alphas, est_error)