import numpy as np
import scipy as sp
import itertools
from scipy.sparse import kron, csc_matrix, lil_matrix, dok_matrix
from functools import reduce

######################
### Gate Functions ###
######################

def qexp(gate, angle):
    """Return matrix exponential of gate"""
    return sp.linalg.expm(- (1j/2) * angle * gate)

def qdot(*args):
    """Return the matrix product of gates"""
    return reduce(np.dot, args).tocsc()

def prod(*args):
    """Returns a product of gates"""
    return reduce(kron, args).tocsc()

def ctrl(*args):
    """Returns a control gate"""
    iictrl = np.asarray([i for i, x in enumerate(args) if type(x) == str])
    ctrlgate = np.asarray(args)
    ctrlgate[iictrl] = [P1]*len(iictrl)
    ctrlgate = prod(*ctrlgate)
    
    template = np.asarray(['I']*len(args), dtype=object)
    offkey = np.asarray(list(itertools.product(('P0', 'P1'), repeat=len(iictrl))), dtype=object)

    for ii in range(0, 2**len(iictrl)-1):
        ctrloff = template
        ctrloff[iictrl] = offkey[ii]
        ctrloff = prod(*[ gdict.get(item,item) for item in ctrloff ])
        ctrlgate = ctrlgate + ctrloff
        
    return ctrlgate

def pTraceM(mat, inds):
    """Returns the partial trace of sparse matrix over list of indices"""
    nrows, ncols = mat.shape[0], mat.shape[1]
    if nrows!=ncols:
        print('Error: non-square matrix')
    elif np.log2(nrows).is_integer() == False:
        print('Error: matrix size not 2^k')
    elif inds[len(inds)-1] > np.log2(nrows) or inds[0] == 0:
        print('Error: index not in (1, nqubit)')
    else:
        inds.sort(reverse = True)
        subcircuit = mat.todok()
    
        for ii in inds:
            nq = int(np.log2(subcircuit.shape[0]))
            traced = dok_matrix((2**(nq - 1), 2**(nq - 1)), dtype=complex)
            if ii == nq:
                for irow in range(traced.shape[0]):
                    for icol in range(traced.shape[1]):
                        traced[irow,icol] = \
                        subcircuit[2*irow:2*irow+2, \
                                   2*icol:2*icol+2].diagonal().sum()
            else:
                dblock = int(np.divide(2**nq, 2**ii))
                for irow in range(0, 2*traced.shape[0] - 1, 2*dblock):
                    for icol in range(0, 2*traced.shape[1] - 1, 2*dblock):
                        traced[int(irow/2):int(irow/2)+dblock, \
                               int(icol/2):int(icol/2)+dblock] = \
                        subcircuit[irow:irow+dblock, \
                                   icol:icol+dblock] + \
                        subcircuit[irow+dblock:irow+2*dblock, \
                                   icol+dblock:icol+2*dblock]  
            subcircuit = traced
        return subcircuit.tocsc()
    
##########################
### Single qubit gates ###
##########################

# Define computational basis for single qubits
q0=csc_matrix([[1],[0]])
q1=csc_matrix([[0],[1]])

# Pauli gates
X = csc_matrix([[0,1],[1,0]])
Y = csc_matrix([[0,-1j],[1j,0]])
Z = csc_matrix([[1,0],[0,-1]])
I = csc_matrix([[1,0],[0,1]])

# Standard gates
H = csc_matrix([[1,1],[1,-1]]).multiply(1/np.sqrt(2))
S=csc_matrix([[1,0],[0,1j]])

# Projection gates
P0 = csc_matrix([[0,0],[0,1]])
P1 = csc_matrix([[1,0],[0,0]])

# Symbolic gates
C='ctrl'

# Money gate

def T1(beta, wx, wz):
    a = np.exp(-0.25*beta*wx)
    b = np.exp(-0.25*beta*wz)
    return np.divide(1, np.sqrt(a**2 + b**2) )*(a*X + b*Z)

H = csc_matrix([[1,1],[1,-1]]).multiply(1/np.sqrt(2))

# Gate dictionary
gdict={'X':X, 'Y':Y, 'Z':Z, 'I':I,
       'P0':P0, 'P1':P1,
       'H':H, 'S':S,
       'ctrl':C}

#######################
### Two-qubit gates ###
#######################

# Basic two-qubit gates
SWAP = csc_matrix([[1,0,0,0],
                   [0,0,1,0],
                   [0,1,0,0],
                   [0,0,0,1]])
FLIP = csc_matrix([[0,0,1,0],
                   [1,0,0,0],
                   [0,0,0,1],
                   [0,1,0,0]])

# Four-dimensional gamma matrices
g2 = [prod(I, X),
      prod(I, Y),
      prod(X, Z),
      prod(Y, Z)]

# Fermionic annihilation gates
c2 = [(1/2)*(g2[2*i] + 1j*g2[2*i + 1]) for i in range(int(len(g2)/2))]

# Left and right-handed SU(2) rotations
Xm = (1/2)*(prod(X, X) + prod(Y, Y))
Ym = (1/2)*(prod(Y, X) - prod(X, Y))
Zm = (1/2)*(prod(Z, I) - prod(I, Z))

Xp = (1/2)*(prod(X, X) - prod(Y, Y))
Yp = (1/2)*(prod(Y, X) + prod(X, Y))
Zp = (1/2)*(prod(Z, I) + prod(I, Z))

Xv, Yv, Zv = Xp + Xm, Yp + Ym, Zp + Zm
Xa, Ya, Za = Xp - Xm, Yp - Ym, Zp - Zm

# Disentangling operations
def Ph2(angle):
    """Returns 2-qubit phase rotation"""
    return qexp( prod(I, I), -2*angle )

def R2(angle):
    """Returns 2-qubit relative rotation"""
    return np.dot( Ph2(angle/2), qexp(Zv, angle) )

def Fou2(p):
    """Returns 2-qubit Fourier gate from scaled momentum p = k/NQ"""
    return reduce(np.dot, [qexp(Ym, np.pi/2), Chi2, R2(2*np.pi*p)] )

def Bog2(angle):
    """Returns 2-qubit Bogoliubov gate"""
    return qexp(Xp, angle)

Chi2 = R2(np.pi)
fSWAP = np.dot( qexp(Ym, np.pi ), Chi2 )

########################
### Four-qubit gates ###
########################

# Sixteen-dimensional gamma matrices
g4 = [prod(I, I, x) for x in g2] + [prod(x, Z, Z) for x in g2]

# Fermionic annihilation gates
c4 = [(1/2)*(g4[2*i] + 1j*g4[2*i + 1]) for i in range(int(len(g4)/2))]

# Qubit rearranger
FLIP4 = prod(SWAP, FLIP)

########################
### Eight-qubit gates ###
########################

# Sixteen-dimensional gamma matrices
g8 = [prod(I, I, I, I, x) for x in g4] + [prod(x, Z, Z, Z, Z) for x in g4]

# Fermionic annihilation gates
c8 = [(1/2)*(g8[2*i] + 1j*g8[2*i + 1]) for i in range(int(len(g8)/2))]
