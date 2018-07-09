import numpy as np
from scipy import signal
import os



def mexican_hat(x, sigma=1.):
    a = 2./ ( np.sqrt(3*sigma) * np.power(np.pi,0.25 ) )
    b = (1. - (x/sigma)**2 )
    c = np.exp( - x**2/(2.*sigma**2))
    return a*b*c


def W_zero(x):
    a          = 1.0
    lambda_net = 4.0
    beta       = 3.0 / lambda_net**2
    gamma      = 1.05 * beta
    
    x_length_squared = x**2
    
    return a*np.exp(-gamma*x_length_squared) - np.exp(-beta*x_length_squared)


def create_W(J, D):
    n = D.shape[0]
    W = np.zeros(D.shape)
    W = J(D) 

    np.fill_diagonal(W, 0.0)
    
    for i in range(n):
        W[i,:] -= np.mean(W[i,:])
    
    return W 


def normalize(x):
    x_   = x - np.amin(x)
    amax = np.amax(x_)

    if amax != 0.:
        x_ = x_/amax
    
    return x_



