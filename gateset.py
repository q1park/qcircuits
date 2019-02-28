import numpy as np
import scipy as sp
import itertools
from scipy.sparse import kron, csc_matrix
from functools import reduce

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
g4 = [prod(I, X),
      prod(I, Y),
      prod(X, Z),
      prod(Y, Z)]

# Left and right-handed SU(2) rotations
Xm = prod(X, X) + prod(Y, Y)
Ym = prod(Y, X) - prod(X, Y)
Zm = prod(Z, I) - prod(I, Z)

Xp = prod(X, X) - prod(Y, Y)
Yp = prod(Y, X) + prod(X, Y)
Zp = prod(Z, I) + prod(I, Z)

Xv, Yv, Zv = Xp + Xm, Yp + Ym, Zp + Zm
Xa, Ya, Za = Xp - Xm, Yp - Ym, Zp - Zm

# Disentangling operations
def Ph2(angle):
    """Returns 2-qubit phase rotation"""
    return qexp( prod(I, I), -angle )

def R2(angle):
    """Returns 2-qubit relative rotation"""
    return np.dot( Ph2(angle), qexp(Zv, angle / 2) )

def Fou2(angle):
    """Returns 2-qubit Fourier gate"""
    return reduce(np.dot, [qexp(Ym, np.pi/4), Chi2, R2(angle)] )

def Bog2(angle):
    """Returns 2-qubit Bogoliubov gate"""
    return qexp(Xp, -angle)

Chi2 = R2(np.pi)
fSWAP = np.dot( qexp(Ym, np.pi / 2), Chi2 )

########################
### Four-qubit gates ###
########################

# Sixteen-dimensional gamma matrices
g8 = [prod(I, I, x) for x in g4] + [prod(x, Z, Z) for x in g4]

# Fermionic annihilation gates
c8 = [(1/2)*(g8[2*i] + 1j*g8[2*i + 1]) for i in range(int(len(g8)/2))]

# Qubit rearranger
FLIP4 = prod(SWAP, FLIP)