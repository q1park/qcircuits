import numpy as np
import scipy as sp
import itertools
from scipy.sparse import kron, csc_matrix, lil_matrix, dok_matrix
from functools import reduce
from gateset import *

###################################################
### Functions for computing XY model parameters ###
###################################################

def S_crit(c, L, l, c1):
    """Calculates the analytical formula for entanglement entropy at criticality"""
    return (c/3.)*np.log2((L/np.pi)*np.sin((l*np.pi)/L ) ) + c1

def w(lam, gam, p):
    """Calculates the single-particle frequencies (p = k/NQ)"""
    return np.sqrt((1 - lam*np.cos(2*np.pi*p ) )**2 + (gam*lam*np.sin(2*np.pi*p ) )**2 )

def theta(lam, gam, p):
    """Calculates the Bogoliubov rotation angle (p = k/NQ)"""
    #lam = lam - 1e-15
    return np.pi - np.arccos(np.divide(-1 + lam*np.cos(2*np.pi*p ), w(lam, gam, p) ) ) 


##########################################
### Functions for testing calculations ###
##########################################

def onequbit_modes(statemat):
    """Returns the Fourier modes of the single-qubit state"""
    nqubit = int(np.log2(statemat.shape[0]))
    rep = np.array(list(itertools.product((0, 1), repeat=nqubit)))
    inds = [i for i, x in enumerate(np.sum(rep, 1)) if x==1]
    
    instates = np.around(statemat[:, inds], 3)

    outstates = np.zeros((len(inds), len(inds)), dtype=complex)
    #print(inds)
    for ii in range(len(inds)):
        shortstate = np.around(instates[sum(instates[:,ii].nonzero()), ii], 3).todense()
        outstates[:, ii] = np.squeeze( np.array( shortstate ) )
        
    return outstates

#######################################################
### Functions for building terms in the Hamiltonian ###
#######################################################

def HXY_iso(gg, sgn):
    nqubit = int( len(gg) )
    HXY = csc_matrix(gg[0].shape, dtype = complex)
    
    for ii in range(nqubit - 1):
        HXY = HXY + qdot(gg[ii].getH(), gg[ii+1] ) + qdot(gg[ii+1].getH(), gg[ii] )
        
    HXY = HXY + sgn*(qdot(gg[nqubit-1].getH(), gg[0] ) + qdot(gg[0].getH(), gg[nqubit-1] ))
    
    return HXY

def HXY_aniso(gg, sgn):
    nqubit = int( len(gg) )
    HXY = csc_matrix(gg[0].shape, dtype = complex)
    
    for ii in range(nqubit - 1):
        HXY = HXY + qdot(gg[ii].getH(), gg[ii+1].getH() ) + qdot(gg[ii+1], gg[ii])
        
    HXY = HXY + sgn*(qdot(gg[nqubit-1].getH(), gg[0].getH() ) + qdot(gg[0], gg[nqubit-1]))
    
    return HXY

def HXY_mag(gg):
    nqubit = int( len(gg) )
    HXY = csc_matrix(gg[0].shape, dtype = complex)
    
    for ii in range(nqubit):
        HXY = HXY + qdot(gg[ii].getH(), gg[ii] )
    
    return HXY

def HXY_diag(gg, evals):
    nqubit = int( len(evals) )
    HXY = csc_matrix(gg[0].shape, dtype = complex)
    norm = (1/2)*csc_matrix(reduce(kron, [I]*nqubit))
    
    for ii in range(len(evals)):
        HXY = HXY + evals[ii]*(qdot(gg[ii].getH(), gg[ii] ) - norm)
    
    return HXY

#############################################################
### Functions for building and calculating various states ###
#############################################################

def fourier_transform(listobj, listp):
    listft = []
    
    for ii in range(len(listp)):
        listweight = [(1/np.sqrt(len(listp) ))*np.exp(2*np.pi*1j*jj*listp[ii]) for jj in range(len(listp))]
        ft = sum([x*y for x,y in zip(listweight, listobj)])
        listft = listft + [ft]
        
    return listft
        

def spin_states(gg):
    nqubit = int( len(gg) )
    #ground = csc_matrix((2**nqubit, 1), dtype = complex)
    statemat = lil_matrix((2**nqubit, 2**nqubit), dtype = complex)
    
    for ii in range(2**nqubit):
        statemat[0,ii] = 1.0
        
    statekey = np.flip(np.array(list(itertools.product((0, 1), repeat=nqubit))), 1)
    
    for ii in range(2**nqubit):
        for jj in range(nqubit):
            if statekey[ii][jj]==1:
                statemat[:, ii] = np.dot(gg[jj].getH(), statemat[:, ii])
    
    return statemat.tocsc()

def build_LP(beta, rw):
    """Builds the Laplacian state from Clifford gates"""
    listgate = []
    
    for iw in range(len(rw)):
        listgate = listgate + [T1(beta, rw[iw], -rw[iw])]

    return prod(*listgate)

def calc_LP(beta, val_E, psi_E):
    """Calculates the Laplacian state analytically"""
    ZP = sum([np.exp(-beta*val_E[x] ) for x in range(val_E.shape[0] ) ] )
    outstate = csc_matrix([np.sqrt(np.divide(np.exp(-beta*val_E[x] ), ZP ) ) \
                           for x in range(val_E.shape[0] ) ]).transpose()
    return outstate

def make_TS(beta, rw):
    """Builds a thermal mixed state"""
    nqubit = int(len(rw) )
    tempLP = build_LP(beta, rw)[:, 0]
    outrho = prod(*([I]*nqubit)).astype(float)

    for it in range(tempLP.shape[0]):
         outrho[it, it] = tempLP[it, 0]
    return outrho

def build_TFD(beta, rw):
    """Builds a thermo-field state from Clifford gates"""
    listprod = []
    nqubit = int(2*len(rw) )
    
    for iw in range(len(rw)):
        icopy = int(iw + len(rw))
        tgate, cgate = [I]*nqubit, [I]*nqubit
        tgate[iw] = T1(beta, rw[iw], -rw[iw])
        cgate[iw] = C
        cgate[icopy] = X
        listprod = listprod + [prod(*tgate), ctrl(*cgate)]
    
    xxgate = [I]*nqubit
    
    for iw in range(len(rw)):
        icopy = int(iw + len(rw))
        xxgate[icopy] = X
        
    listprod = listprod + [prod(*xxgate)]
    listprod = list(reversed(listprod) )
    
    return qdot(*listprod)

def calc_TFD(beta, val_E, psi_E):
    """Calculates the thermo-field double state analytically"""
    ZP = sum([np.exp(-beta*val_E[x] ) for x in range(val_E.shape[0] ) ] )
    outstate = csc_matrix((len(val_E)**2, 1 ), dtype=complex)
    
    for ii in range(len(val_E)):
        coeff = np.sqrt(np.divide(np.exp(-beta*val_E[ii]), ZP ))
        outstate = outstate + coeff*prod(psi_E[:,ii], psi_E[:,ii])

    return outstate

def scansig_time(Utime, state, Udis, gates, wider = False):
    """Calculates the expectation value of 1 or 2 Pauli operators in space and time"""
    typekey = 0
    
    if state.shape[0]==state.shape[1]:
        typekey = 1
        
    initgate = []
    scangate = csc_matrix((2, 2), dtype=complex)
    
    if len(gates)==1:
        initgate, scangate = [I], gates[0]
    elif len(gates)==2:
        initgate, scangate = [gates[0]], gates[1]
    else:
        print('invalid gate selection')
        
    nqubit = int(np.log2(state.shape[0] ) )
    scanout = np.zeros((len(Utime), nqubit))
    
    for itp in range(len(Utime) ):
        timestate = csc_matrix((2, 2), dtype=complex)
        
        if typekey==0:
            phases = np.matrix(Utime[itp].diagonal() ).transpose()
            timestate = np.dot(Udis, state.multiply(phases))
        if typekey==1:
            timestate = reduce(np.dot,[Udis, Utime[itp], 
                                       state, 
                                       Utime[itp].getH(), Udis.getH()])
        
        for ix in range(int(nqubit) ):
            imat = initgate + [I]*(nqubit - 1)
            imat[ix] = scangate
            expval = 0.0
            
            if typekey==0:
                expval = reduce(np.dot,[timestate.getH(), prod(*imat), timestate])[0,0]
            if typekey==1:
                expval = np.array(np.dot(prod(*imat), timestate).diagonal() ).sum()
                if np.imag(expval)>0.01:
                    print(expval)
                
            scanout[itp, ix] = expval.astype(float)

    if wider==True:
        scanout_wide = np.zeros((len(Utime), 2*nqubit))
        
        for ix in range(int(nqubit) ):
            scanout_wide[:, 2*ix] = scanout[:, ix]
            scanout_wide[:, 2*ix+1] = scanout[:, ix]
        scanout = scanout_wide
    return np.flip(scanout, 1)

def scansig_temp(arrb, rw, Udis, gates, type = 'single'):
    """Calculates the expectation value 
    of 1 or 2 Pauli operators for thermal 
    states in space and temperature"""
    
    sdict = {'double':0, 'single':1, 'mixed':2}
    typekey = sdict.get(type)
    
    initgate = []
    scangate = csc_matrix((2, 2), dtype=complex)
    
    if len(gates)==1:
        initgate, scangate = [I], gates[0]
    elif len(gates)==2:
        initgate, scangate = [gates[0]], gates[1]
    else:
        print('invalid gate selection')
    
    nqubit = int(len(rw) )
    
    if typekey==0:
        nqubit = int(2*len(rw) )
        
    scanout = np.zeros((len(arrb), nqubit))
    
    for ib in range(len(arrb) ):
        tempstate = csc_matrix((2, 2), dtype=complex)
        
        if typekey==0:
            tempstate = np.dot(Udis, build_TFD(arrb[ib], rw)[:, 0])
        if typekey==1:
            tempstate = np.dot(Udis, build_LP(arrb[ib], rw)[:, 0])
        if typekey==2:
            tempstate = reduce(np.dot, [Udis, make_TS(arrb[ib], rw), Udis.getH()] )
        
        for ix in range(int(nqubit) ):
            imat = initgate + [I]*(nqubit - 1)
            imat[ix] = scangate
            expval = 0.0
            
            if typekey==0 or typekey==1:
                expval = reduce(np.dot,[tempstate.getH(), prod(*imat), tempstate])[0, 0]
            if typekey==2:
                expval = np.array(np.dot(prod(*imat), tempstate).diagonal() ).sum()
                
            scanout[ib, ix] = expval.astype(float)
    return np.flip(scanout, 0)

def scanent_temp(arrb, rw, Udis, type = 'single'):
    """Calculates the entanglement entropy for thermal 
    states in space and temperature"""
    
    sdict = {'double':0, 'single':1, 'mixed':2}
    typekey = sdict.get(type)
    
    nqubit = int(len(rw) )
    
    if typekey==0:
        nqubit = int(2*len(rw) )
        
    scanout = np.zeros((len(arrb), nqubit))
    
    for ib in range(len(arrb) ):
        tempstate = csc_matrix((2, 2), dtype=complex)
        
        if typekey==0:
            tempstate = np.dot(Udis, build_TFD(arrb[ib], rw)[:, 0])
            tempstate = np.dot(tempstate, tempstate.getH() )
        if typekey==1:
            tempstate = np.dot(Udis, build_LP(arrb[ib], rw)[:, 0])
            tempstate = np.dot(tempstate, tempstate.getH() )
        if typekey==2:
            tempstate = reduce(np.dot, [Udis, make_TS(arrb[ib], rw), Udis.getH()] )
        
        for ix in reversed(range(int(nqubit) ) ):
            
            tempstate = pTraceM(tempstate, [ix+1])
            #print(type(tempstate), type(sp.linalg.logm(tempstate)) )
            #print(-np.dot(tempstate.todense(), sp.linalg.logm(tempstate.todense() ) ))
            entval = -np.dot(tempstate.todense(), 
                             sp.linalg.logm(tempstate.todense() ) / np.log(2) ).diagonal().sum()
                
            scanout[ib, ix] = entval.astype(float)
    return np.flip(scanout, 0)

def scanent_time(Utime, state, Udis):
    """Calculates the entanglement entropy for pure 
    states in space and temperature"""
    
    if state.shape[0]!=state.shape[1]:
        print('Error: must input a square density matrix')
    
    nqubit = int(np.log2(state.shape[0] ) )
    scanout = np.zeros((len(Utime), nqubit))
            
    for itp in range(len(Utime) ):
        
        tempstate = csc_matrix((2, 2), dtype=complex)
        tempstate = reduce(np.dot,[Udis, Utime[itp], 
                                   state, 
                                   Utime[itp].getH(), Udis.getH()])
        
        for ix in reversed(range(int(nqubit) ) ):
            #print(ix, np.around(tempstate.todense(),4))
            tempstate = pTraceM(tempstate, [ix+1])
            
            entval = -np.dot(tempstate.todense(), 
                             sp.linalg.logm(tempstate.todense() ) / np.log(2) ).diagonal().sum()
            scanout[itp, ix] = entval.astype(float)
            
    return scanout