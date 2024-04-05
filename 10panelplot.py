import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import cmasher as cmr

from plotting import (
    make_FMR_fig, linear, fourth_order
)

mpl.rcParams['font.size'] = 50
mpl.rcParams['axes.linewidth']    = 2.25*3.5
mpl.rcParams['xtick.major.width'] = 1.5 *3.5
mpl.rcParams['ytick.major.width'] = 1.5 *3.5
mpl.rcParams['xtick.minor.width'] = 1.0 *3.5
mpl.rcParams['ytick.minor.width'] = 1.0 *3.5
mpl.rcParams['xtick.major.size'] = 8   * 3.5
mpl.rcParams['ytick.major.size'] = 8   * 3.5
mpl.rcParams['xtick.minor.size'] = 4.5 * 3.5
mpl.rcParams['ytick.minor.size'] = 4.5 * 3.5

sims = ['original','tng','eagle']

savedir = './Figures (pdfs)/'
STARS_OR_GAS = "gas".upper()

for all_z_fit in [True, False]:
    for function in [linear]:#, fourth_order]:
        for sim in sims:
            make_FMR_fig(sim, all_z_fit, savedir=savedir, 
                         function=function)