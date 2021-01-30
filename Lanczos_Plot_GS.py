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

#plt.close('all')
hf = hm.FermionicBasis_1d(3, 3, 6)

U  = 5.0
t1 = time.perf_counter()
m  = 0
min_qL = np.zeros_like(hf.momenta)

for i,qx in enumerate(hf.momenta):
  H = hm.H_Qx(hf, qx, U)
  states, eig, Ndone, eps = hm.Davidson(H.tocsr(), 50, m)
  #eig = eigsh(H,k=5,sigma=0.,return_eigenvectors=False,which='SM')
  min_qL[i] = min(eig)

t2 = time.perf_counter()
print(f"Exact diagonalization in {t2-t1}s")
print(f"Ground state energy: { min_qL.min() }")

plt.rc('text',usetex=True)
cpalette = sns.color_palette("icefire",n_colors = len(hf.momenta)+1)

# Just for symmetry
plot_momenta = hf.momenta.copy()
plot_momenta = np.append(plot_momenta, -hf.momenta[0])

plt.figure(figsize=(14,7))
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.xlabel(r"$q$", fontsize = 26)
plt.ylabel(r"$\min E_q$", fontsize = 26)
plt.gca().tick_params(axis='both', which='major', labelsize=14)

min_qL = np.append(min_qL, min_qL[0])

# Smooth-line in the Thermodynamic Limit
min_qL_TL = interpolate.interp1d(plot_momenta, min_qL,kind='quadratic')
q_TL     = np.linspace(plot_momenta[0],plot_momenta[-1],1000)

plt.plot(q_TL, min_qL_TL(q_TL), '--', color = 'k', linewidth=0.75,zorder=1)
plt.scatter(plot_momenta, min_qL, c=cpalette, s=60,zorder=2)
plt.xticks(plot_momenta)

