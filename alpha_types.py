import os
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from matplotlib.colors import LogNorm

from helpers import (
    get_all_redshifts, WHICH_SIM_TEX, get_z0_alpha
)
from plotting import (
    linear
)

def get_delta_evo(sim):
    sim = sim.upper()
    z0_mbins, z0_zbins, z0_Sbins = None, None, None
    z0_SFMS, z0_MZR = None, None
    
    unique = np.arange(0,9)
    
    delta_evo = np.zeros(len(unique))
    evo_min = np.zeros(len(unique))
    evo_max = np.zeros(len(unique))
    
    min_alpha, *_ = get_z0_alpha(sim,function=linear)
    
    for index, redshift in enumerate(unique):
        mbins, zbins, Sbins = get_avg_relations(sim,redshift)
        
        if index == 0:
            z0_mbins, z0_zbins, z0_Sbins = mbins, zbins, Sbins
            z0_mu = z0_mbins - min_alpha * np.log10(z0_Sbins)
            params = np.polyfit(z0_mu, z0_zbins, 1)
            
            z0_zpred = linear(z0_mu, *params)
            
            z0_SFMS = interp1d(z0_mbins, z0_Sbins, fill_value='extrapolate')
            z0_MZR = interp1d(z0_mbins, z0_zpred, fill_value='extrapolate')
            continue
        
        deltas = np.linspace(0,1,200)
        MSE = np.zeros(len(deltas))
        
        for j, delta in enumerate(deltas):
            Z_pred = -delta * np.log10( Sbins / z0_SFMS(mbins) ) + z0_MZR(mbins)
            MSE[j] = sum((zbins - Z_pred)**2) / len(Z_pred)
        
        delta_evo[index] = deltas[np.argmin(MSE)]
        
        loss = np.log10(MSE)
        
        width = np.min(loss) * 0.95
        
        within_uncertainty = deltas[ (loss < width) ]

        evo_min[index] = within_uncertainty[0]
        evo_max[index] = within_uncertainty[-1]
        
    return delta_evo, evo_min, evo_max

def delta_evo_plot(sim):
    mpl.rcParams['font.size'] = 26
    fig = plt.figure(figsize=(8,6))
    ax  = plt.gca()
    
    redshifts_to_plot = [1,3,8]
    
    sim = sim.upper()
    z0_mbins, z0_zbins, z0_Sbins = None, None, None
    z0_SFMS, z0_MZR = None, None
    
    unique = np.arange(0,9)

    cmap = cmr.get_sub_cmap('cmr.guppy', 0.0, 1.0, N=len(unique))
    newcolors = np.linspace(0, 1, len(unique))
    colors = [ cmap(x) for x in newcolors[::-1] ]
    lCol = []
    
    delta_evo = np.zeros(len(unique))
    evo_min = np.zeros(len(unique))
    evo_max = np.zeros(len(unique))
    
    ls = ['solid','dotted','dashdot']
    line_styles = ['' for _ in range(len(unique))]
    j = 0
    for index in redshifts_to_plot:
        line_styles[index] = ls[j]
        j+=1
    
    min_alpha, *_ = get_z0_alpha(sim,function=linear)
    
    for index, redshift in enumerate(unique):
        mbins, zbins, Sbins = get_avg_relations(sim,redshift)
        
        if index == 0:
            z0_mbins, z0_zbins, z0_Sbins = mbins, zbins, Sbins
            z0_mu = z0_mbins - min_alpha * np.log10(z0_Sbins)
            params = np.polyfit(z0_mu, z0_zbins, 1)
            
            z0_zpred = linear(z0_mu, *params)
            
            z0_SFMS = interp1d(z0_mbins, z0_Sbins, fill_value='extrapolate')
            z0_MZR = interp1d(z0_mbins, z0_zpred, fill_value='extrapolate')
            continue ## delta @ z=0 is, by definition, 0
        
        deltas = np.linspace(-0.1,1.1,400)
        MSE = np.zeros(len(deltas))
        
        for j, delta in enumerate(deltas):
            Z_pred = -delta * np.log10( Sbins / z0_SFMS(mbins) ) + z0_MZR(mbins)
            MSE[j] = sum((zbins - Z_pred)**2) / len(Z_pred)
        
        delta_evo[index] = deltas[np.argmin(MSE)]
        
        loss = np.log10(MSE)
        
        width = np.min(loss) * 0.95
        within_uncertainty = deltas[ (loss < width) ]
        
        evo_min[index] = within_uncertainty[0]
        evo_max[index] = within_uncertainty[-1]

        if index in redshifts_to_plot:
            color = colors[index]
            lCol.append(color)
            L = loss - np.min(loss)
            L /= np.max(L)
            
            # width -= np.min(loss)
            # width /= np.max(L)
            ax.plot(deltas, L, lw=3, color=color,label='$z=%s$'%index,
                    linestyle=line_styles[index])
            ax.scatter(delta_evo[index],0,color=color,s=100)
            for line in [evo_min[index],evo_max[index]]:
                ax.scatter(line,0.05,color=color,marker='s',s=100)
        
    ax.axhline(0.0,color='k',lw=3)
    ax.axhline(0.05,color='gray',linestyle='--')
    ax.text(0,-0.01,r'${\rm MSE~Minimum}$',
            va='top',ha='left',color='k',fontsize=20)
    ax.text(0,0.05,r'$5\%~{\rm deviation}$',
            va='bottom',ha='left',color='gray',fontsize=20)
    
    ax.set_yticks([])
    
    ax.text(1.01,0.1,WHICH_SIM_TEX[sim],ha='right')
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.09,0.75)
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'$\log {\rm MSE}~({\rm scaled~units})$')
    leg=ax.legend(frameon=False,loc='center right',labelspacing=0.05)
    for n, text in enumerate( leg.texts ):
        text.set_color( lCol[n] )
    plt.tight_layout()
    plt.savefig('./Figures (pdfs)/' + 'delta_demo' + '.pdf', bbox_inches='tight')
        
def polynomial(mass, SFR, metallicity, order=1, nbins=25):
    
    mbins = np.linspace(np.min(mass),np.max(mass),nbins)
    zbins = np.polyval(np.polyfit(mass, metallicity, order), mbins)
    Sbins = np.polyval(np.polyfit(mass, SFR, 1), mbins)
    
    return mbins, zbins, Sbins

def median(mass, metallicity, SFR, nbins=25):
    ''' DEPRECATED '''
    mbins = np.linspace(np.min(mass),np.max(mass),nbins)
    zbins = np.full_like(mbins, fill_value=np.nan)
    Sbins = np.full_like(mbins, fill_value=np.nan)
    
    width = (np.max(mass) - np.min(mass)) / nbins

    for i, current in enumerate(mbins):
        mask = (mass > current) & (mass < (current + width))
        if np.sum(mask) > 10:
            zbins[i] = np.median(metallicity[mask])
            Sbins[i] = np.median(SFR[mask])
    
    no_nans = ~(np.isnan(zbins)) & ~(np.isnan(Sbins))
    
    return mbins[no_nans], zbins[no_nans], Sbins[no_nans]

def get_medians(x,y,z,width=0.05,step=0.1,min_samp=15):
    start = np.min(x)
    end   = np.max(x)
    
    xs = np.arange(start,end,step)
    median_y = np.zeros( len(xs) )
    median_z = np.zeros( len(xs) )
    
    for index, current in enumerate(xs):
        mask = ((x > (current)) & (x < (current + width)))
        
        if (len(y[mask]) > min_samp):
            median_y[index] = np.median(y[mask])
            median_z[index] = np.median(z[mask])
        else:
            median_y[index] = np.nan
            median_z[index] = np.nan
        
    nonans = ~(np.isnan(median_y)) & ~(np.isnan(median_z))
    
    xs = xs[nonans] + step/2
    median_y = median_y[nonans]
    median_z = median_z[nonans]

    return xs, median_y, median_z

def get_avg_relations(sim,redshift):
    star_mass, SFR, Z, redshifts = get_all_redshifts(sim, False)
    
    mask = redshifts == redshift
    current_m = star_mass[mask]
    current_Z = Z[mask]
    current_S = SFR[mask]

    return get_medians(current_m, current_Z, current_S)

def line(x, a, b):
    return a*x + b

def fourth_order(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

if __name__ == "__main__":
    print('Hello World')