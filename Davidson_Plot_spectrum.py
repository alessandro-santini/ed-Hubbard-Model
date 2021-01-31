# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:15:52 2021

@author: asant
"""

import HubbardModelTools as hm
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
import time 
import seaborn as sns
from scipy import interpolate
# from scipy.sparse.linalg import eigsh

plt.close('all')
hf = hm.FermionicBasis_1d(5, 5, 10)

U  = 5.0
t1 = time.perf_counter()
m  = 5

Dav_spectrum = np.zeros((hf.momenta.size,m))

for i,qx in enumerate(hf.momenta):
  H = hm.H_Qx(hf, qx, U)
  states, eig, Ndone, eps = hm.Davidson(H.tocsr(), 1000, m)
  Dav_spectrum[i, :] = eig[:m]

t2 = time.perf_counter()
print(f"Exact diagonalization in {t2-t1}s")
print(f"Ground state energy: { np.min(Dav_spectrum) }")

plt.figure(figsize=(14,7))
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.rc('text',usetex=True)
cpalette = sns.color_palette("icefire",n_colors = len(hf.momenta)+1)

plot_momenta = hf.momenta.copy()
plot_momenta = np.append(plot_momenta, -hf.momenta[0])
q_TL     = np.linspace(plot_momenta[0],plot_momenta[-1],1000)

plt.xlabel(r"$q$", fontsize = 26)
plt.ylabel(r"$\min E_q$", fontsize = 26)
plt.gca().tick_params(axis='both', which='major', labelsize=14)

for energies in Dav_spectrum.T:    
    en_plot = np.append(energies, energies[0])
    # Smooth-line in the Thermodynamic Limit
    en_plot_TL = interpolate.interp1d(plot_momenta, en_plot,kind='quadratic')
    plt.plot(q_TL, en_plot_TL(q_TL), '--', color = 'k', linewidth=0.75,zorder=1)
    plt.scatter(plot_momenta, en_plot, c=cpalette, s=50,zorder=2)
    plt.xticks(plot_momenta)