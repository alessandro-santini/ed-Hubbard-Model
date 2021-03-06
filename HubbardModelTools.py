import numpy as np
import numpy.ma as ma
import scipy.sparse as scsp
import scipy.linalg as sl
import time

class FermionicBasis_1d:
    def __init__(self, Nup, Ndown, L):
        self.Nup = Nup # Number of spin up electrons
        self.Ndown = Ndown # Number of spin down electrons
        self.N = Nup + Ndown # Total Number of electrons
        self.L = L # Number of lattice 
        self.momenta = np.arange(-L//2, L//2)*2*np.pi/L
        # Saves the parities of the fermionic system
        self.Pup    = Nup   % 2
        self.Pdown  = Ndown % 2
        # Representative states of the system
        print(f"Building the Hubbard model basis with L={L}, Nup={Nup} and Ndown={Ndown}")
        t1  = time.perf_counter()
        rep = self.build_representative_states()
        ## Orders and saves in an attribute the representatives
        self.representatives = rep[np.lexsort((rep[:, 1], rep[:, 0]))]
        t2 = time.perf_counter()
        print(f"Time spent to build the basis {t2-t1}s")
    
    def build_all_states(self):
        """ Returns all the possible states of the Hubbard model 
            with a 1d lattice of length L and Nup,Ndown electrons """
        basis_up = set()
        basis_down = set()
        for i in np.arange(0,2**self.L):
          b = np.binary_repr(i, width=self.L)
          if(b.count('1') == self.Nup):
            basis_up.add(i)
          if(b.count('1') == self.Ndown):
            basis_down.add(i)
        up_array   = np.array(list(basis_up))
        down_array = np.array(list(basis_down))
        states = np.transpose([np.tile(up_array, len(down_array)), np.repeat(down_array, len(up_array))])    
        return states
    
    def build_representative_states(self):
        All_states = set() # <-- this contains all the state that are generated by a cycle which we are already counting
        Repres_states = np.array([[0, 0]]) # Creates a null ``numpy list''
        for state in self.build_all_states():
          if ( str( state ) in All_states): continue
          cycle_state = np.array([state])
          for i_x in np.arange(1, self.L):
            cycle_state = np.append(cycle_state,  np.array([list( map( lambda z: (np.left_shift(z,i_x) % (2**self.L -1)  ) , state) )]), axis=0 )
          cycle_state = np.unique(cycle_state, axis=0)
          
          # Adds the states to the set in a hashable way
          All_states.update( list(map(str, cycle_state)))
          
          # Index of the minimum of the cycled state
          index_min_up = np.argmin(cycle_state[:,0])
      
          mask_array = np.ones((len(cycle_state)), dtype=int)
          mask_array[index_min_up] = 0
          masked_array = ma.masked_array(cycle_state[:,1], mask_array)
      
          min_index = np.where(masked_array.min() == masked_array)
          Repres_states = np.append(Repres_states, cycle_state[min_index], axis=0)
        return np.delete(Repres_states, 0, axis=0)
    
    def ComputeNorm_Qx(self, qx):
        """Returns the norm of the representative states"""
        NormStatesQx = np.zeros(len(self.representatives))
        for r, rep_state in enumerate(self.representatives):      
          for i_x in np.arange(1, self.L+1):    
            shifted_rep = np.array( list( map( lambda z: ( np.left_shift(z, i_x) % (2**self.L -1) ), rep_state) ) ) 
            if ( (shifted_rep == rep_state).all() ): # <-- You need less than L translations to go back to the original state
              period = i_x
              break
          sign = (-1)**(np.binary_repr(rep_state[0], self.L)[:period].count('1')*(self.Nup-1)+\
                        np.binary_repr(rep_state[1], self.L)[:period].count('1')*(self.Ndown-1))
          
          if (period == self.L):
            NormStatesQx[r] = np.sqrt(self.L)
          else:
            Fourier_components = np.zeros(period, dtype = np.complex128)
            for i_x in np.arange(0, self.L):
              j_x = i_x % period
              i_sign = ( i_x // period )
              Fourier_components[j_x] += np.exp(1j*i_x*qx)*sign**i_sign
            NormStatesQx[r] = np.linalg.norm(Fourier_components)
        return NormStatesQx
    
    def set_RepQx(self, qx):
        norm_qx = self.ComputeNorm_Qx(qx)
        self.RepQx     = self.representatives[norm_qx>1e-10]
        self.SetRepQx  = set(list(map(str, self.RepQx)))
        self.NormRepQx = norm_qx[norm_qx>1e-10]

    def check_rep(self, UpInt, DownInt):
        """ Returns the representative state of the state [UpInt,DownInt]
            the number of translations j_x needed link these states and the sign of this transformation """
        info = False
        for j_x in np.arange(0, self.L):
          Trial_Rep = np.array([[ np.left_shift(UpInt, j_x) , np.left_shift(DownInt, j_x) ]]) % (2**self.L -1)
          if (str(Trial_Rep[0]) in self.SetRepQx):
            info = True
            sign = (-1)**( np.binary_repr(UpInt, self.L)[:j_x].count('1')*(self.Nup-1) +\
                          np.binary_repr(DownInt,self.L)[:j_x].count('1') * (self.Ndown-1) )
            return Trial_Rep, j_x, sign, info
        return [0,0], 0, 0, info

# Builds the block of the Hamiltonian
def swap(s, i, j):
    """Returns the state obtained by swapping the bits (i, j) of the state s"""
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)
    
def H_Qx(basis, qx, U):
    """ Return the block with momentum qx of the Hamiltonian """
    basis.set_RepQx(qx)
        
    len_RepQx = basis.RepQx.shape[0]
    RepQxToIndex = dict(zip(list(map(str,basis.RepQx)), np.arange(0, len_RepQx))) 
    H = scsp.dok_matrix((len_RepQx, len_RepQx), dtype=np.complex128)

    for Index_rep, rep in enumerate(basis.RepQx):
        
        # Binary representation of the states for the representative Index_rep
        Up_state   = np.binary_repr(rep[0], width=basis.L)
        Down_state = np.binary_repr(rep[1], width=basis.L)
    
        for i_x in np.arange(0, basis.L):
          inext_x = (i_x+1) % basis.L # Nearest neighbour with pbc
          #Diagonal term contribution
          if(Up_state[i_x]=='1' and Down_state[i_x]=='1'):
            H[Index_rep, Index_rep] += U
    
          #Acting c^\dagger c on up spin state
          if(Up_state[i_x]=='0' and Up_state[inext_x]=='1'):
            NewUpInt = int(swap(Up_state, i_x, inext_x), 2) 
            Swapped_rep, j_x, sign, info = basis.check_rep(NewUpInt, rep[1])
            if(info):
                Index_Swapped_rep = RepQxToIndex[str(Swapped_rep[0])]
                sign = sign*(1-2*(i_x==basis.L-1))**(basis.Nup-1)
                H[Index_rep, Index_Swapped_rep] -= np.exp( 1j*qx*j_x) * basis.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]*sign
                H[Index_Swapped_rep, Index_rep] -= np.exp(-1j*qx*j_x) * basis.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]*sign
    
          #Acting c^\dagger c on down spin state
          if(Down_state[i_x]=='0' and Down_state[inext_x]=='1'):
            NewDownInt = int(swap(Down_state, i_x, inext_x), 2) 
            Swapped_rep, j_x, sign, info = basis.check_rep(rep[0],NewDownInt)
            if(info):   
                Index_Swapped_rep = RepQxToIndex[str(Swapped_rep[0])]
                sign=sign*(1-2*(i_x==basis.L-1))**(basis.Ndown-1)
                H[Index_rep, Index_Swapped_rep] -= np.exp( 1j*qx*j_x) * basis.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]*sign
                H[Index_Swapped_rep, Index_rep] -= np.exp(-1j*qx*j_x) * basis.NormRepQx[Index_Swapped_rep]/basis.NormRepQx[Index_rep]*sign
    return H

def Lanczos(H,Psi,Nsteps,m=4,eps=1e-8):
  """Diagonalize the lowest m eigenvalues with a tolerance of eps in the
     residual, i.e. ||A V_m s -\lambda A V_m s||  """
  N = Psi.shape[0]
  Psi = Psi/np.linalg.norm(Psi)
  PsiMinus = np.zeros_like(Psi, dtype=np.complex128)
  PsiPlus  = np.zeros_like(Psi, dtype=np.complex128)
  # Eigenvectors matrix
  Vm    = np.reshape(Psi.copy(),newshape=(N,1))
  alpha = np.array([])
  beta  = np.array([])

  alpha = np.append(alpha, np.vdot(Psi,H.dot(Psi)) )
  beta = np.append(beta,0.0)
  for i in range(1,Nsteps+1):
    PsiPlus  = (H.dot(Psi)-alpha[i-1]*Psi)-beta[i-1]*PsiMinus
    beta     = np.append(beta,np.linalg.norm(PsiPlus))
    PsiPlus  = PsiPlus/beta[i]
    Vm       = np.append(Vm,np.reshape(PsiPlus,newshape=(N,1) ),axis=1)
    PsiMinus = Psi.copy()
    Psi      = PsiPlus.copy()

    alpha  = np.append(alpha, np.vdot(Psi,H.dot(Psi)) )
    eig, s = sl.eigh_tridiagonal(alpha.real,beta[1:].real)
    if(i > m):
      u = np.matmul(Vm, s[:,m])
      r = ( H.dot(u)-eig[m]*u )
      if (np.linalg.norm(r)<eps):
        break
  return np.matmul(Vm,s), eig, i, np.linalg.norm(r)

def Davidson(H,Nsteps,m=4,eps=1e-8):
  
  N = H.shape[0]
  Psi = np.random.random(N) + 1j*np.random.random(N)
  Psi = Psi/np.linalg.norm(Psi)
  PsiMinus = np.zeros(N,dtype=np.complex128)
  PsiPlus  = np.zeros(N,dtype=np.complex128)

  Vm=np.reshape(Psi.copy(),newshape=(N,1))
  alpha = np.array([])
  beta  = np.array([])

  alpha = np.append(alpha, np.vdot(Psi,H.dot(Psi)) )
  beta = np.append(beta,0.0)

  m_trial=0
  for i in range(1,Nsteps+1):

    if (i<=m): #Lanczos if i<m
      PsiPlus  = (H.dot(Psi)-alpha[i-1]*Psi)-beta[i-1]*PsiMinus
      beta     = np.append(beta,np.linalg.norm(PsiPlus))
      PsiPlus  = PsiPlus/beta[i]
      
      Vm_Trial = np.append(Vm, np.reshape(PsiPlus,newshape=(N,1) ),axis=1)
      
      PsiMinus = Psi.copy()
      Psi      = PsiPlus.copy()
      alpha    = np.append(alpha, np.vdot(Psi,H.dot(Psi)) )
      eig, s   = sl.eigh_tridiagonal(alpha.real,beta[1:].real)
      Vm = Vm_Trial.copy()

    else:

      H_small = np.matmul(Vm.T.conjugate(),H @ Vm)
      eig, s   = np.linalg.eigh(H_small)

      u = np.matmul(Vm, s[:, m_trial])

      r = ( H.dot(u)- eig[m_trial]*u )
      
      if (np.linalg.norm(r) < eps ):
        if (m==m_trial):
          break
        else:
          m_trial +=1
          u = np.matmul(Vm, s[:, m_trial])
          r = ( H.dot(u)- eig[m_trial]*u )
          t =  -r/(H.diagonal() - eig[m])# np.linalg.solve(np.matmul(P_perp,H.dot(P_perp)), -r/(H.diagonal() - eig[m] ) ) 
          for qi in Vm.T:
              t -= np.vdot(qi, t)*qi 
          Vm = np.append(Vm, np.reshape(t/np.linalg.norm(t),newshape = (N,1) ),axis=1)
      # Davidson Construction of the vector
      else:
        
          # P_perp = scsp.eye(u.shape[0]) - np.tensordot(u.conjugate(), u, axes=0)
          t =  -r/(H.diagonal() - eig[m])# np.linalg.solve(np.matmul(P_perp,H.dot(P_perp)), -r/(H.diagonal() - eig[m] ) ) 
          for qi in Vm.T:
              t -= np.vdot(qi, t)*qi 
          Vm = np.append(Vm, np.reshape(t/np.linalg.norm(t),newshape = (N,1) ),axis=1)
    
  return np.matmul(Vm,s), eig, i, np.linalg.norm(r)