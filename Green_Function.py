import HubbardModelTools as hm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from   scipy import interpolate

def c(s, i):
    lst = list(s)
    if(lst[i]=='0'): raise Exception("Error: passing a state annihilated by c")
    lst[i] = '0'
    return ''.join(lst)

def c_q_up(basis,basis_minus,state,qx,k):
    len_RepQx_minus = len(basis_minus.RepQx)
    RepQxToIndex_minus = dict(zip(list(map(str,basis_minus.RepQx)), np.arange(0, len_RepQx_minus))) 
    components = np.zeros(len_RepQx_minus, dtype = np.complex128)    
    for Index_rep, rep in enumerate(basis.RepQx):
            if (np.abs(state[Index_rep])<10**-15): continue
            Up_state   = np.binary_repr(rep[0], width = basis.L)
            for i in np.arange(0,hf.L):
                if(Up_state[i] == '1'):
                    NewUpInt = int(c(Up_state,i), 2)
                    Swapped_rep, j_x, sign, info = basis_minus.check_rep(NewUpInt, rep[1])
                    sign = sign*(-1)**np.binary_repr(NewUpInt,width = basis.L)[:i].count('1')
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_minus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                            state[Index_rep]*basis_minus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
    return components

hf   = hm.FermionicBasis_1d(3, 3, 6)
U    = 5.
H    = hm.H_Qx(hf,0.,U) 
dimH = H.shape[0]
v0   = np.random.random(dimH)+1j*np.random.random(dimH)
states, eig, Ndone, _ = hm.Lanczos(H,v0,200,m=0)

gs_energy = eig[0]
gs_state  = states[:,0]

hf_minus = hm.FermionicBasis_1d(2, 3, 6)
H_minus  = hm.H_Qx(hf_minus, 0., U)

components = c_q_up(hf,hf_minus,gs_state,0.)
