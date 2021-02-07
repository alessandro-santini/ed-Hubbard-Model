import HubbardModelTools as hm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from   scipy import interpolate
import scipy.linalg as sl
from scipy.signal import find_peaks

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
                    sign = sign*(-1)**(np.binary_repr(NewUpInt,width = basis.L)[:i].count('1')+np.binary_repr(rep[1],width = basis.L)[:i].count('1'))
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_minus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                            state[Index_rep]*basis_minus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
    return components/np.linalg.norm(components)

def n_q(basis,basis_minus,state,k,qx):
    len_RepQx_minus = len(basis_minus.RepQx)
    RepQxToIndex_minus = dict(zip(list(map(str,basis_minus.RepQx)), np.arange(0, len_RepQx_minus))) 
    components = np.zeros(len_RepQx_minus, dtype = np.complex128)    
    for Index_rep, rep in enumerate(basis.RepQx):
            if (np.abs(state[Index_rep])<10**-15): continue
            if( not( str(rep) in RepQxToIndex_minus)): continue
            Index_n_rep = RepQxToIndex_minus[str(rep)]
            Up_state   = np.binary_repr(rep[0], width = basis.L)
            Down_state   = np.binary_repr(rep[1], width = basis.L)
            for j in np.arange(0,hf.L):
                #By keeping only up/down one gets the operator for only up/down densities
                Nup   = int(Up_state[j])
                Ndown = int(Down_state[j])
                components[Index_n_rep] += state[Index_rep]*(Nup+Ndown)*np.exp(-1j*qx*j)*basis_minus.NormRepQx[Index_n_rep]/basis.NormRepQx[Index_rep]
    return components/np.linalg.norm(components)


hf   = hm.FermionicBasis_1d(4, 4, 8)
#For C_q
#hf_minus = hm.FermionicBasis_1d(3, 4, 8)
#For N_q
hf_minus = hm.FermionicBasis_1d(4, 4, 8)

#Better check those before every run
U    = 0.
k    = np.pi
H    = hm.H_Qx(hf,k,U)
dimH = H.shape[0]
v0   = np.random.random(dimH)+1j*np.random.random(dimH)
states, eig, Ndone, _ = hm.Lanczos(H,v0,100,m=0)
gs_energy = eig[0]
gs_state  = states[:,0]


n_lanc = 50
n_g = 1000
G = np.zeros(n_g)
zspace = np.linspace(gs_energy,20+gs_energy,n_g)
epsi = 1j*1e-10


#Before running check the following: k,q,Operator,hf_minus
for iii,q in enumerate(hf.momenta):
    
    H_minus = hm.H_Qx(hf_minus,k-q,U)

####Lanczos procedure for density Green's function####
    
    N = len(hf_minus.RepQx)
    #For C_q
    #Psi = c_q_up(hf,hf_minus,gs_state,q,k)
    #For N_q
    Psi = n_q(hf,hf_minus,gs_state,q,k)

    PsiMinus = np.zeros_like(Psi, dtype=np.complex128)
    PsiPlus  = np.zeros_like(Psi, dtype=np.complex128)

    Vm    = np.reshape(Psi.copy(),newshape=(N,1))
    alpha = np.array([])
    beta  = np.array([])
    alpha = np.append(alpha, np.vdot(Psi,H_minus.dot(Psi)) )
    beta  = np.append(beta,0.0)

    for i in np.arange(1,n_lanc):

        PsiPlus  = (H_minus.dot(Psi)-alpha[i-1]*Psi)-beta[i-1]*PsiMinus
        beta     = np.append(beta,np.linalg.norm(PsiPlus))
        PsiPlus  = PsiPlus/beta[i]
        Vm       = np.append(Vm,np.reshape(PsiPlus,newshape=(N,1) ),axis=1)
        PsiMinus = Psi.copy()
        Psi      = PsiPlus.copy()
        
        alpha  = np.append(alpha, np.vdot(Psi,H_minus.dot(Psi)) )

    u = np.zeros(shape=(n_lanc,1),dtype=np.complex128)
    u[0,0]=1.

    for iz,z in enumerate(zspace):
        m = np.diag(z+epsi-alpha, k=0)-np.diag(beta[1:],k=1)-np.diag(beta[1:].conjugate(),k=-1) 
        num = np.linalg.det( np.append(u,m[:,1:],axis=1) )
        den = np.linalg.det(m)
        G[iz] += (num/den).imag


print(zspace[find_peaks(abs(G))[0]])
plt.plot(zspace,abs(G))
plt.yscale('log')
plt.show()




"""
    #Lanczos procedure for density Green's function
    N = len(hf.RepQx)
    Psi = n_q0(hf,gs_state)
    PsiMinus = np.zeros_like(Psi, dtype=np.complex128)
    PsiPlus  = np.zeros_like(Psi, dtype=np.complex128)

    Vm    = np.reshape(Psi.copy(),newshape=(N,1))
    alpha = np.array([])
    beta  = np.array([])
    alpha = np.append(alpha, np.vdot(Psi,H.dot(Psi)) )
    beta = np.append(beta,0.0)

    for i in np.arange(1,100):

        PsiPlus  = (H.dot(Psi)-alpha[i-1]*Psi)-beta[i-1]*PsiMinus
        beta     = np.append(beta,np.linalg.norm(PsiPlus))
        PsiPlus  = PsiPlus/beta[i]
        Vm       = np.append(Vm,np.reshape(PsiPlus,newshape=(N,1) ),axis=1)
        PsiMinus = Psi.copy()
        Psi      = PsiPlus.copy()

        alpha  = np.append(alpha, np.vdot(Psi,H.dot(Psi)) )
        eig, s = sl.eigh_tridiagonal(alpha.real,beta[1:].real)
    u = np.zeros(shape=(100,1),dtype=np.float64)
    u[0,0]=1.

    G = np.zeros(100)
    zspace=np.linspace(0,10,100)
    for iz,z in enumerate(zspace):
        m = np.diag(z-alpha, k=0)+np.diag(beta[1:],k=1)+np.diag(beta[1:],k=-1) 
        num = np.linalg.det( np.append(u,m[:,1:],axis=1) )
        den = np.linalg.det(m)
        G[iz] = (num/den).imag

    plt.plot(zspace,G)
    plt.show()
"""
