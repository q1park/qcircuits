import numpy as np
import copy

def loss_func(umat1, umat2):
    reUsq = np.real(umat1 - umat2).toarray()**2
    imUsq = np.imag(umat1 - umat2).toarray()**2
    return np.sqrt(np.sum(reUsq) + np.sum(imUsq))

def specsum(cc, iden, hh):
    output = cc[0]*iden
    
    for i in range(len(hh)):
        output = output + cc[i+1]*hh[i]
        
    return output

def grad_comp(truth, ind, cc, iden, hh):
    epsR = 0.001
    epsI = 0.001*1j
    
    ccupR, ccdnR = copy.deepcopy(cc), copy.deepcopy(cc)
    ccupI, ccdnI = copy.deepcopy(cc), copy.deepcopy(cc)
    
    ccupR[ind] = ccupR[ind] + epsR
    ccdnR[ind] = ccdnR[ind] - epsR
    
    ccupI[ind] = ccupI[ind] + epsI
    ccdnI[ind] = ccdnI[ind] - epsI
    
    lossupR = loss_func(truth, specsum(ccupR, iden, hh))
    lossdnR = loss_func(truth, specsum(ccdnR, iden, hh))
    
    lossupI = loss_func(truth, specsum(ccupI, iden, hh))
    lossdnI = loss_func(truth, specsum(ccdnI, iden, hh))
    
    gradR = np.divide(lossupR - lossdnR, 2*np.absolute(epsR))
    gradI = np.divide(lossupI - lossdnI, 2*np.absolute(epsI))
    
    return gradR + 1j*gradI

def update_spec(truth, cc, iden, hh):
    lr = 0.001
    ccnew = copy.deepcopy(cc)
    
    for iicc in range(len(ccnew)):
        gradient = grad_comp(truth, iicc, cc, iden, hh)
        ccnew[iicc] = ccnew[iicc] - lr*gradient
        
    return ccnew