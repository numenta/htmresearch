import numpy as np
from scipy import signal
import os





def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def save_data(data, filename, expid):
    for key in data:
        with open(filename.format(expid, key), "w") as f:
            np.save(f, data[key])



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


def optical_flow(I1g, I2g, pixels, w=1):
    
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])


    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) +\
         signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)

    for i,j in pixels:
    # for i in range(w, I1g.shape[0]-w, 5):
        # for j in range(w, I1g.shape[1]-w, 5):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            A = np.zeros((2,2))
            b = np.zeros((2,1))
            A[0,0] = np.sum(Ix**2)
            A[0,1] = np.sum(Iy*Ix)
            A[1,0] = np.sum(Iy*Ix)
            A[1,1] = np.sum(Iy**2)
            b[0,0] = - np.sum(Ix*It)
            b[1,0] = - np.sum(Iy*It)
            A_ = np.linalg.pinv(A)
            nu = np.dot(A_, np.dot(A.T,b) )
            u[i,j]=nu[0,0]
            v[i,j]=nu[1,0]
 

    return (u,v)

def cw(theta):
    s = theta / (2.*np.pi)

    if s <= 1./3.:
        s_ = s*3.
        return (1-s_,s_,0)
    if s <= 2./3.:
        s_ = (s - 1./3.)*3.
        return (0,1-s_,s_)
    if s <= 3./3.:
        s_ = (s - 2./3.)*3.
        return (s_,0,1-s_)
    


def flow_to_color(u,v):
    n = len(u)
    
    c = np.zeros((n, 3))
    theta = np.angle(u + v*1j) % (2.*np.pi)

    norm = np.sqrt(u**2 + v**2)

    for i in range(n):
        c[i] = cw(theta[i])

    return c


def add_padding(arr, val=0., w=1):
    arr[  :w,   : ] = val
    arr[-w: ,   : ] = val
    arr[  : ,   :w] = val
    arr[  : , -w: ] = val

def get_active_pixels(mask):
    return np.array(np.where(mask==1.)).reshape((2,-1)).T

def get_data_flow_and_color_maps(S, nx, ny, pixel_mask, t_step = 10):

    T, n = S.shape
    S_ = S[np.arange(0, T, step=t_step)]
    S_ = S_.reshape((-1,nx,ny))     # Subsample of activity array

    T_ = S_.shape[0]
    C  = np.zeros((T_, n, 3))       # RGB Color array
    V  = np.zeros((T_, nx, ny, 2))  # Flow velocity
    V_ = np.zeros((T_, nx, ny, 2))  # Processed velocity


    w = 1
    add_padding(pixel_mask, 0., w)
    pixels = get_active_pixels(pixel_mask)

    for t in np.arange(T_  - 10):
        vx, vy = optical_flow(S_[t], S_[t + 10], pixels, w)
        V[t,:,:,0] = vx
        V[t,:,:,1] = vy


    w = 4
    for t in np.arange(T_  - 10):

        for i in np.arange(w, nx-w):
            for j in np.arange(w, ny-w):
                W = V[t, i-w:i+w, j-w:j+w].reshape((-1,2))
                normW = np.linalg.norm(W, axis=1)
                max_ind = np.argmax(normW)
                V_[t, i, j] = W[max_ind]

    V_ = V_.reshape((-1,n,2))
    for t in range(T_  - 10):
        C[t] = flow_to_color(V_[t,:,0],V_[t,:,1])  



    C = C.reshape((T_,nx,ny,3))
    C[:,:w,:,:]  = 1.
    C[:,-w:,:,:] = 1.
    C[:,:,:w,:]  = 1.
    C[:,:,-w:,:] = 1.
    # C[:, pixel_mask==0.] = 0.

    C = C.reshape((T_,n,3))
    S_ = S_[:T_ - 10].reshape((-1,n))
    V  = V [:T_ - 10]
    C  = C [:T_ - 10]
    V_ = V_[:T_ - 10]

    return S_, V_, C
    



def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



