import numpy as np


def relu(x):
    return np.maximum(x, 0.)


def evolve_step(W, b, s, beta=0., mask=1.):
    
    dt  = 0.01
    tau = 1.0
    f   = relu
    n   = W.shape[0]

    
    ds  = dt * ( f(np.dot(W,s) + b + beta) - s/tau )
    s_  = s + ds
    s_ *= mask

    if np.sum(s_**2) >0:
            s_ = s_/np.sqrt(np.sum(s_**2))
            
    return s_



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

