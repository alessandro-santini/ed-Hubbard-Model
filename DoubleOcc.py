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
hf = hm.FermionicBasis_1d(3, 3, 6)

N_u = 101
Uspace = np.linspace(0,-50,N_u,endpoint=True)
D_occ_u = np.zeros(N_u,dtype=np.float128)
for i_u, U in enumerate(Uspace):
  t1 = time.perf_counter()
  m  = 1
  e_gs=100000.
  Dav_spectrum = np.zeros((hf.momenta.size,m))

  for i,qx in enumerate(hf.momenta):
    H = hm.H_Qx(hf, qx, U)
    states, eig, Ndone, eps = hm.Davidson(H.tocsr(), 1000, m)
    if(eig[0]<e_gs): 
      gs_state=states[:,0]
      e_gs = eig[0]
      base = hf.RepQx.copy()
      Qx_gs = qx
    Dav_spectrum[i, :] = eig[:m]

  N_double = 0.
  LL = hf.L
  for i_rep, rep in enumerate(base):
    #if(gs_state[i_rep]<1e-14):continue
    UpInt = np.binary_repr(rep[0],LL)
    DownInt = np.binary_repr(rep[1],LL)
    for ii in range(LL):
      if (UpInt[ii]=='1' and DownInt[ii]=='1'):
        N_double += 1*abs(gs_state[i_rep])**2
  N_double = N_double/LL
  print(f"GS double count:{N_double}")
  t2 = time.perf_counter()
  print(f"Exact diagonalization in {t2-t1}s")
  print(f"Ground state energy: { np.min(Dav_spectrum) }")
  D_occ_u[i_u] = N_double

plt.figure()
plt.ylabel(f"Nd")
plt.xlabel(f"U")
plt.plot(Uspace,D_occ_u)
plt.show()

