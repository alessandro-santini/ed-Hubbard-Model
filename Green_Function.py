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

def cdag(s, i):
    lst = list(s)
    if(lst[i]=='1'): raise Exception(r"Error: passing a state annihilated by c^\dagger")
    lst[i] = '1'
    return ''.join(lst)


#C_q
def c_q_up(basis,basis_minus,state,qx,k):
    len_RepQx_minus = len(basis_minus.RepQx)
    RepQxToIndex_minus = dict(zip(list(map(str,basis_minus.RepQx)), np.arange(0, len_RepQx_minus))) 
    components = np.zeros(len_RepQx_minus, dtype = np.complex128)    
    for Index_rep, rep in enumerate(basis.RepQx):
            if (np.abs(state[Index_rep])<10**-15): continue
            Up_state   = np.binary_repr(rep[0], width = basis.L)
            for i in np.arange(0,basis.L):
                if(Up_state[i] == '1'):
                    NewUpInt = int(c(Up_state,i), 2)
                    Swapped_rep, j_x, sign, info = basis_minus.check_rep(NewUpInt, rep[1])
                    sign = sign*(-1)**(np.binary_repr(NewUpInt,width = basis.L)[:i].count('1')+np.binary_repr(rep[1],width = basis.L)[:i].count('1'))
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_minus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                            state[Index_rep]*basis_minus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
    return components/np.linalg.norm(components)

def c_q_down(basis,basis_minus,state,qx,k):
    len_RepQx_minus = len(basis_minus.RepQx)
    RepQxToIndex_minus = dict(zip(list(map(str,basis_minus.RepQx)), np.arange(0, len_RepQx_minus))) 
    components = np.zeros(len_RepQx_minus, dtype = np.complex128)    
    for Index_rep, rep in enumerate(basis.RepQx):
            if (np.abs(state[Index_rep])<10**-15): continue
            Down_state   = np.binary_repr(rep[1], width = basis.L)
            for i in np.arange(0,basis.L):
                if(Down_state[i] == '1'):
                    NewDownInt = int(c(Down_state,i), 2)
                    Swapped_rep, j_x, sign, info = basis_minus.check_rep(rep[0], NewDownInt)
                    sign = sign*(-1)**(np.binary_repr(NewDownInt,width = basis.L)[:i].count('1')+np.binary_repr(rep[0],width = basis.L)[:i].count('1'))
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_minus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                            state[Index_rep]*basis_minus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
    return components/np.linalg.norm(components)


#C^dagger_q
def cdag_q_up(basis,basis_plus,state,qx,k):
    len_RepQx_plus = len(basis_plus.RepQx)
    RepQxToIndex_plus = dict(zip(list(map(str,basis_plus.RepQx)), np.arange(0, len_RepQx_plus))) 
    components = np.zeros(len_RepQx_plus, dtype = np.complex128)    
    for Index_rep, rep in enumerate(basis.RepQx):
            if (np.abs(state[Index_rep])<10**-15): continue
            Up_state   = np.binary_repr(rep[0], width = basis.L)
            for i in np.arange(0,basis.L):
                if(Up_state[i] == '0'):
                    NewUpInt = int(cdag(Up_state,i), 2)
                    Swapped_rep, j_x, sign, info = basis_plus.check_rep(NewUpInt, rep[1])
                    sign = sign*(-1)**(np.binary_repr(NewUpInt,width = basis.L)[:i].count('1')+np.binary_repr(rep[1],width = basis.L)[:i].count('1'))
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_plus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                            state[Index_rep]*basis_plus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
    return components/np.linalg.norm(components)

def cdag_q_down(basis,basis_plus,state,qx,k):
    len_RepQx_plus = len(basis_plus.RepQx)
    RepQxToIndex_plus = dict(zip(list(map(str,basis_plus.RepQx)), np.arange(0, len_RepQx_plus))) 
    components = np.zeros(len_RepQx_plus, dtype = np.complex128)    
    for Index_rep, rep in enumerate(basis.RepQx):
            if (np.abs(state[Index_rep])<10**-15): continue
            Down_state   = np.binary_repr(rep[1], width = basis.L)
            for i in np.arange(0,basis.L):
                if(Down_state[i] == '1'):
                    NewDownInt = int(c(Down_state,i), 2)
                    Swapped_rep, j_x, sign, info = basis_plus.check_rep(rep[0], NewDownInt)
                    sign = sign*(-1)**(np.binary_repr(NewDownInt,width = basis.L)[:i].count('1')+np.binary_repr(rep[0],width = basis.L)[:i].count('1'))
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_plus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                            state[Index_rep]*basis_plus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
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
            for j in np.arange(0,basis.L):
                #By keeping only up/down one gets the operator for only up/down densities
                Nup   = int(Up_state[j])
                Ndown = int(Down_state[j])
                components[Index_n_rep] += state[Index_rep]*(Nup+Ndown)*np.exp(-1j*qx*j)*basis_minus.NormRepQx[Index_n_rep]/basis.NormRepQx[Index_rep]
    return components/np.linalg.norm(components)

# Current <jG^-1j>
# j_x = c^\dagger_i *( c_{i-1} - c_{i+1})
# j_x = c^dagger_i  c_{i-1} - c^\dagger_i c_{i+1}
# i-1 ----> i   +
# i   <---- i+1 -
# j_q = \sum_{n} e^{iqn} j_n
def j_q_up(basis,basis_minus,state,k,qx):
    len_RepQx_minus = len(basis_minus.RepQx)
    RepQxToIndex_minus = dict(zip(list(map(str,basis_minus.RepQx)), np.arange(0, len_RepQx_minus))) 
    components = np.zeros(len_RepQx_minus, dtype = np.complex128)    
    for Index_rep, rep in enumerate(basis.RepQx):
            if (np.abs(state[Index_rep])<10**-15): continue
            Up_state   = np.binary_repr(rep[0], width = basis.L)
            for i in np.arange(0,basis.L):
                iprev = (i+1)%basis.L
                inext = (i-1)%basis.L
                if(Up_state[i] == '1'): continue
                # Right hop ___  c^\dagger_i c_{i-1}
                if(Up_state[iprev]=='1'): 
                    NewUpInt = int( cdag(c(Up_state,iprev), i), 2)
                    Swapped_rep, j_x, sign, info = basis_minus.check_rep(NewUpInt, rep[1])
                    if(i==0):
                        sign = sign*(-1)**(basis.N+1)
                    # else: not get a sign
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_minus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += 1j*sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                        state[Index_rep]*basis_minus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
                # Left hop  ___ -c^\dagger_i c_{i+1}
                if(Up_state[inext]=='1'): 
                    NewUpInt = int( cdag(c(Up_state,inext), i), 2)
                    Swapped_rep, j_x, sign, info = basis_minus.check_rep(NewUpInt, rep[1])
                    if(i==basis.L):
                        sign = sign*(-1)**(basis.N)
                    # else: not get a sign
                    if(info):
                        Index_Swapped_rep = RepQxToIndex_minus[str(Swapped_rep[0])]
                        components[Index_Swapped_rep] += 1j*sign*np.exp( 1j*(j_x*(k-qx)-qx*i) )*\
                        state[Index_rep]*basis_minus.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]
    return components/np.linalg.norm(components)    

hf   = hm.FermionicBasis_1d(4, 4, 8)
#For C_q
#hf_minus = hm.FermionicBasis_1d(3, 4, 8)
#For N_q
hf_minus = hm.FermionicBasis_1d(4, 4, 8)

#Better check those before every run
for ijk,U in enumerate(np.linspace(6,12,13,endpoint=True)):
    k    = np.pi
    H    = hm.H_Qx(hf,k,U)
    dimH = H.shape[0]
    v0   = np.random.random(dimH)+1j*np.random.random(dimH)
    m_state = 1
    states, eig, Ndone, _ = hm.Lanczos(H,v0,100,m=m_state)
    gs_energy = eig[m_state]
    gs_state  = states[:,m_state]
    
    
    n_lanc = 50
    n_g = 4000
    G = np.zeros(n_g)
    wspace = np.linspace(-20,20,n_g)
    zspace = gs_energy+wspace
    epsi = 1j*1e-1
    
    
    #Before running check the following: k,q,Operator,hf_minus
    for iii,q in enumerate([0.0]):
        
        H_minus = hm.H_Qx(hf_minus,k-q,U)
    
    ####Lanczos procedure for density Green's function####
        
        N = len(hf_minus.RepQx)
        #For C_q
        #Psi = c_q_up(hf,hf_minus,gs_state,q,k)
        #For N_q
        Psi = j_q_up(hf,hf_minus,gs_state,q,k)
    
        PsiMinus = np.zeros_like(Psi, dtype=np.complex128)
        PsiPlus  = np.zeros_like(Psi, dtype=np.complex128)
    
        Vm    = Psi.copy().reshape(N,1)
        alpha = np.array([])
        beta  = np.array([])
        alpha = np.append(alpha, np.vdot(Psi,H_minus.dot(Psi)) )
        beta  = np.append(beta,0.0)
    
        for i in np.arange(1,n_lanc):
    
            PsiPlus  = (H_minus.dot(Psi)-alpha[i-1]*Psi)-beta[i-1]*PsiMinus
            beta     = np.append(beta,np.linalg.norm(PsiPlus))
            PsiPlus  = PsiPlus/beta[i]
            Vm       = np.append(Vm,PsiPlus.reshape(N,1),axis=1)
            PsiMinus = Psi.copy()
            Psi      = PsiPlus.copy()
            
            alpha  = np.append(alpha, np.vdot(Psi,H_minus.dot(Psi)) )
    
        u = np.zeros(shape=(n_lanc,1),dtype=np.complex128)
        u[0,0]=1.
    
        for iw,w in enumerate(wspace):
            m = np.diag(zspace[iw]+epsi-alpha, k=0)-np.diag(beta[1:],k=1)-np.diag(beta[1:].conjugate(),k=-1) 
            B_num = m.copy() #np.linalg.det( np.append(u,m[:,1:],axis=1) )
            B_num[:,0] =  u[:,0]
            num = np.linalg.det(B_num)   
            den = np.linalg.det(m)
            G[iw] += (num/den).imag
    
    G = -G/hf.N/abs(wspace)
    
    print(zspace[find_peaks(abs(G))[0]])
    peaks = find_peaks(abs(G))[0]
    plt.plot(wspace[len(wspace)//2:], G[:len(wspace)//2][::-1] + G[len(wspace)//2:])
    plt.title("U: %.3f"%(U))
    plt.yscale('log')
    #plt.ylim(-1,1)
    plt.show()
    #plt.plot(zspace[peaks]-gs_energy,((G/(zspace-gs_energy+1e-7))[peaks]))
    #plt.savefig("./figure/%d.png"%(ijk), format='png', dpi=600 )
    #plt.close('all')
    
    
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
