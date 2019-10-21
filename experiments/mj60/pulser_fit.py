import pandas as pd
import sys
import numpy as np
import scipy as sp
import json
import os
from decimal import Decimal
import scipy.optimize as opt
from scipy.optimize import minimize, curve_fit
from scipy.special import erfc
from scipy.stats import crystalball
from scipy.signal import medfilt, find_peaks
import pygama.analysis.histograms as pgh
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('style.mplstyle')

    
if(len(sys.argv) != 2):
    print('Usage: fit_bkg_peaks.py [run number]')
    sys.exit()

with open("runDB.json") as f:
    runDB = json.load(f)
tier_dir = os.path.expandvars(runDB["tier_dir"])

df =  pd.read_hdf("{}/t2_run{}.h5".format(tier_dir,sys.argv[1]))

def gauss(x, mu, sigma, A):
    """
    define a gaussian distribution, w/ args: mu, sigma, area (optional).
    """
    return A * (1. / sigma / np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2. * sigma**2))

hist, bins, var = pgh.get_hist(df['e_ftp'], range=(7065,7085), dx=0.5)
pgh.plot_hist(hist, bins, var=hist, label="data")
pars, cov = pga.fit_hist(gauss, hist, bins, var=hist, guess=[7078, 5, 3566])
pgu.print_fit_results(pars, cov, gauss)
pgu.plot_func(gauss, pars, label="chi2 fit", color='red')
#x_vals = np.arange(345,360,0.5)
#plt.plot(x_vals, radford_peak(x_vals, 353, 1.05, .001, 0.02, 500, 1000, 40000))

FWHM = '%.2f' % Decimal(pars[1]*2)
FWHM_uncertainty = '%.2f' % Decimal(np.sqrt(cov[1][1])*2)

#chi_2_element_list = []
#for i in range(len(hist)):
    #chi_2_element = abs((radford_peak(bins[i], *pars) - hist[i])**2/radford_peak(bins[i], *pars))
    #chi_2_element_list.append(chi_2_element)
#chi_2 = sum(chi_2_element_list)
#reduced_chi_2 = '%.2f' % Decimal(chi_2/len(hist))

label_01 = 'pulser peak fit'
label_02 = 'FWHM = '+str(FWHM)+r' $\pm$ '+str(FWHM_uncertainty)
colors = ['red', 'red']
lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
labels = [label_01, label_02]

plt.xlim(7065,7085)
plt.ylim(0,plt.ylim()[1])
plt.xlabel('ADC', ha='right', x=1.0)
plt.ylabel('Counts', ha='right', y=1.0)

plt.tight_layout()
#plt.semilogy()
plt.legend(lines, labels, frameon=False, loc='upper right', fontsize='small')
plt.show()

