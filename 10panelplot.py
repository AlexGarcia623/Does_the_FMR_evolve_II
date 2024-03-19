import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import cmasher as cmr

from plotting import make_FMR_fig

mpl.rcParams['font.size'] = 50
mpl.rcParams['xtick.major.width'] = 1.5 *2.5
mpl.rcParams['ytick.major.width'] = 1.5 *2.5
mpl.rcParams['xtick.minor.width'] = 1.0 *2.5
mpl.rcParams['ytick.minor.width'] = 1.0 *2.5
mpl.rcParams['xtick.major.size']  = 7.5 *2.5
mpl.rcParams['ytick.major.size']  = 7.5 *2.5
mpl.rcParams['xtick.minor.size']  = 3.5 *2.5
mpl.rcParams['ytick.minor.size']  = 3.5 *2.5
mpl.rcParams['axes.linewidth']    = 2.25*2.5

sims = ['original','tng','eagle']

savedir = './Figures (pdfs)/'
STARS_OR_GAS = "gas".upper()

### This makes Figures 1, 3, and A1-4
for all_z_fit in [False, True]:
    for sim in sims:
        make_FMR_fig(sim, all_z_fit, savedir=savedir)