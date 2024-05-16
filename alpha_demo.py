import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from plotting import (
    make_alpha_evo_plot
)

mpl.rcParams['font.size'] = 24
mpl.rcParams['axes.linewidth'] = 5
mpl.rcParams['xtick.major.width'] = 2.75
mpl.rcParams['ytick.major.width'] = 2.75
mpl.rcParams['xtick.minor.width'] = 1.75
mpl.rcParams['ytick.minor.width'] = 1.75
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 7.5
mpl.rcParams['ytick.minor.size'] = 7.5

make_alpha_evo_plot('TNG',redshifts = [5])