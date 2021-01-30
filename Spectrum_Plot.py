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

hf = hm.FermionicBasis_1d(4, 4, 8)

U = 5.0
t1 = time.perf_counter()

eig_qx = np.array([])
qx_array = np.array([]) 
eps=1e-8
m = 5

for qx in hf.momenta:
  H = hm.H_Qx(hf, qx, U)
  H = H.toarray()
  eigs     = np.sort(np.linalg.eigh(H)[0])
  eig_qx   = np.concatenate([eig_qx, eigs])
  qx_array = np.concatenate([qx_array, qx*np.ones_like(eigs) ])

t2 = time.perf_counter()
spectrum = np.array([eig_qx, qx_array]).transpose()
print(f"Exact diagonalization in {t2-t1}s")
print(f"Ground state energy: { spectrum.min() }")


plt.rc('text',usetex=True)

# Just for symmetry
plot_momenta = hf.momenta.copy()
plot_momenta = np.append(plot_momenta, -hf.momenta[0])

plt.figure(figsize=(14,7))
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

cpalette = sns.color_palette("icefire",n_colors = len(hf.momenta)+1)
for c, qx in enumerate(hf.momenta):
    plt.plot(spectrum[:,1][spectrum[:,1]==qx],spectrum[:,0][spectrum[:,1]==qx],'o',color = cpalette[c], markersize=3)

# Just for symmetry
plt.plot(-spectrum[:,1][spectrum[:,1]==hf.momenta[0]],spectrum[:,0][spectrum[:,1]==hf.momenta[0]],'o',color = cpalette[-1], markersize=3)

plt.xticks(plot_momenta)
plt.xlabel("$q$", fontsize = 26)
plt.ylabel("$E_q$", fontsize = 26)
plt.gca().tick_params(axis='both', which='major', labelsize=14)

plt.figure(figsize=(14,7))
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

min_q = np.array([min(spectrum[:,0][spectrum[:,1]==qx]) for qx in hf.momenta])
plt.xlabel(r"$q$", fontsize = 26)
plt.ylabel(r"$\min E_q$", fontsize = 26)
plt.gca().tick_params(axis='both', which='major', labelsize=14)

min_q = np.append(min_q, min_q[0])

# Smooth-line in the Thermodynamic Limit
min_q_TL = interpolate.interp1d(plot_momenta, min_q,kind='quadratic')
q_TL     = np.linspace(plot_momenta[0],plot_momenta[-1],1000)

plt.plot(q_TL, min_q_TL(q_TL), '--', color = 'k', linewidth=0.75,zorder=1)
plt.scatter(plot_momenta, min_q, c=cpalette, s=60,zorder=2)
plt.xticks(plot_momenta)

plt.show()