#%%
import HubbardModelTools as hm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from   scipy import interpolate

hf = hm.FermionicBasis_1d(5, 5, 10)
U  = 5.
H  = hm.H_Qx(hf,0.,U) 
dimH = H.shape[0]
v0 = np.random.random(dimH)+1j*np.random.random(dimH)
states, eig, _, _ = hm.Lanczos(H,v0,100,m=3)
gs_energy = eig[0]
gs_state  = states[:,0]

hf_minus = hm.FermionicBasis_1d(5,4,10)
hf_minus.set_RepQx(0.)
#%%
