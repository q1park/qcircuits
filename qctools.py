import numpy as np
import pandas as pd
import sympy as sy
import unittest

from functools import reduce
from scipy import linalg
from scipy.sparse import kron, csc_matrix, dok_matrix#, lil_matrix
#from sympy import radsimp, sqrt, init_printing

class CircuitError(Exception):
    """Circuit exception class"""
    pass

class SizeError(CircuitError):
    """Circuit exception class"""
    def __init__(self, message):
        self.message = message

class ShapeError(CircuitError):
    """Circuit exception class"""
    def __init__(self, message):
        self.message = message

class CircuitTools(object):
    def __init__(self, circuit):
        """Class to create, modify, and manipulate a quantum circuit"""
        self.circuit = circuit
        self.nrows = self.circuit.shape[0]
        self.ncols = self.circuit.shape[1]
        
    def pTrace(self, inds):
        """Returns the partial trace of sparse matrix over list of indices"""
        try:
            if self.nrows!=self.ncols:
                raise ShapeError('Error: non-square matrix')
            if np.log2(self.nrows).is_integer() == False:
                raise SizeError('Error: matrix size not 2^k')
            if inds[len(inds)-1] > np.log2(self.nrows) or inds[0] == 0:
                raise CircuitError('Error: index not in (1, nqubit)')
            
            inds.sort(reverse = True)
            subcircuit = self.circuit.todok()
            
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
        
        except (CircuitError, SizeError, ShapeError) as err:
            print(err)
    
    def evolve_state(self, state_vector):
        """Adds a new layer of gates to the existing circuit"""
        return np.dot(self.circuit, state_vector)
    
    def set_zero(self):
        """Prints circuit as pandas dataframe"""
        tempmat = np.around( self.circuit, 3)
        tempmat.eliminate_zeros()
        return csc_matrix(tempmat)
    
    def check_recover(self, state_vector):
        """Checks if the first qubit in a state tensor-factorizes"""
        logicalone=state_vector.todok()[:int(len(state_vector.todok().toarray())/2)]
        logicalzero=state_vector.todok()[int(len(state_vector.todok().toarray())/2):]

        if logicalone.nnz > 0 and logicalzero.nnz > 0:
            return(print("qubit 1 = not factorizable"))
        elif logicalone.nnz==0:
            return print("qubit 1 = logical one")
        elif logicalzero.nnz==0:
            return print("qubit 1 = logical zero")
        #return (logicalzero!=logicalone).nnz==0
    
    def check_nnz(self):
        """Prints circuit as pandas dataframe"""
        return (self.set_zero()).nnz
    
    def prettymatrix(self):
        """Prints circuit as pandas dataframe"""
        prettymat = np.array(self.set_zero().todense())
        return pd.DataFrame([[c if c.imag else c.real for c in b] for b in np.around(prettymat, 2)])
    
    def prettystate(self, state_vector):
        """Prints state as pandas dataframe"""
        state_np = np.array( np.around( state_vector.todense(), 3))
        return pd.DataFrame([[c if c.imag else c.real for c in b] for b in state_np])
    
    def prettyqubit(self, state_vector):
        """Converts a vector into a tensor product sum in computational basis"""
        nq = np.log2(len(state_vector.toarray())).astype(int)
        terms = np.asarray(sy.symbols(':2'*nq))
        state_np = np.around(state_vector.toarray(), 3)
        
        pretty = sum(t*c for t, c in zip(terms, state_np))

        if type(pretty) == np.ndarray:
            return pretty[0]
        else: 
            return pretty
    
if __name__ == "__main__":
    unittest.main()