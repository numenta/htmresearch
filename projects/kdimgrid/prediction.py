import numpy as np
from scipy.stats import skewnorm

def fit_params_to_1d_data(logX):
    """
    Fit skewed normal distributions to 1-D capactity data, 
    and return the distribution parameters.

    Args
    ----
    logX:
        Logarithm of one-dimensional capacity data, 
        indexed by module and phase resolution index

    """
    m_max = logX.shape[0]
    p_max = logX.shape[1]
    params = np.zeros((m_max, p_max, 3)) 
    for m_ in range(m_max):
        for p_ in range(p_max):
            params[m_,p_] = skewnorm.fit(logX[m_,p_])
    
    return params


def get_interpolated_params(m_frac, ph, params):    
    """
    Get parameters describing a 1-D capactity distribution 
    for fractional number of modules.
    """
    slope, offset = np.polyfit(np.arange(1,4), params[:3,ph,0], deg=1)
    a   = slope*m_frac + offset
    
    slope, offset = np.polyfit(np.arange(1,4), params[:3,ph,1], deg=1)
    loc   = slope*m_frac + offset
    
    slope, offset = np.polyfit(np.arange(1,4), params[:3,ph,2], deg=1)
    scale   = slope*m_frac + offset

    return (a, loc, scale)


def predict_log_capacity(m,n,p, logX, params, raw=False):

    m_frac = float(m)/float(n)        
        
    a, loc, scale = get_interpolated_params(m_frac, p, params)
    x_min = loc - 5.*scale
    x_max = loc + 5.*scale
    x = np.linspace(x_min, x_max, num=100)
    
    f = skewnorm.pdf(x, a, loc, scale)
    f = f/np.sum(f)
    C = np.random.choice(x, p=f, size=(n,100000))

    pred_raw  = np.amin(C, axis=0)
    pred_mean = np.mean(pred_raw) 

    return pred_raw if raw else pred_mean


def construct_predictions(logX, ms, ks, ps, raw=False):
    params     = fit_params_to_1d_data(logX)
    if raw:
        prediction = np.zeros((max(ms), max(ks), len(ps), 100000))
    else:
        prediction = np.zeros((max(ms), max(ks), len(ps)))
    
    # 
    #   UNDERFLOW WARNING:
    #   
    #   The function `skewnorm` within `predict_log_capacity` can cause an 
    #   underflow warning/error, but it doesn't seem to be an issue for our computations. 
    #   The function f in `predict_log_capacity` looks fine even 
    #   after normalization. So I just ignore the error :|
    #
    old_err_setting = np.seterr(under="warn")

    for m,k,p in [ (m,k,p) for m in ms for k in ks for p in ps]:
        m_ = m-1
        k_ = k-1
        if m >= k:
            pred = predict_log_capacity(m,k,p, logX, params, raw)
            prediction[m_,k_,p] = np.exp(pred)

    np.seterr(**old_err_setting)
    
    return prediction



  
