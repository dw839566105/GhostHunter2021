import tables
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from zernike import RZern
from tqdm import tqdm
import h5py

order1 = 10
order2 = 10
@jit(nopython=True)
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x

def calc_basis(order1, order2, r, cos_theta, t):
    # Zernike basis

    cart = RZern(order1)
    nk = cart.nk
    m = cart.mtab
    n = cart.ntab
    theta = np.arccos(cos_theta)
    rr = np.repeat(r, cos_theta.shape[0]/r.shape[0])
    X = np.zeros((rr.shape[0], nk))
    for i in np.arange(nk):
        if not i % 5:
            print(f'process {i}-th event')
        X[:,i] = cart.Zk(i, rr, theta)
    X = X[:,m>=0]
    # Legendre basis

    # T = legval(2*t/310-1, np.eye(order2).reshape((order2,order2,1))).T
    T = legval(t, np.eye(order2).reshape((order2,order2,1))).T
    X_total = np.repeat(X,T.shape[0],axis=0)
    global order_eff
    order_eff = X_total.shape[1]
    T_total = np.tile(T,(X.shape[0],1))
    #t_rep = np.tile(2*t/310-1, X.shape[0])
    t_rep = np.tile(t, X.shape[0])
    r_rep = np.repeat(rr,T.shape[0])
    
    # interaction_pairs
    basis = np.zeros((X_total.shape[0], X_total.shape[1] * T_total.shape[1]))
    for index1 in tqdm(np.arange(X_total.shape[1])):
        for index2 in np.arange(T_total.shape[1]):
            basis[:, index2 + index1*T_total.shape[1]] = X_total[:,index1] * T_total[:,index2]
    return basis, t_rep, r_rep
N = 100
xx,yy = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100), sparse=False)

r = np.sqrt(xx.flatten()**2 + yy.flatten()**2)
cos_theta = yy.flatten()/(r+1e-6)
t = np.linspace(-1,1,100)
basis, t_rep, r_rep = calc_basis(order1, order2, r, cos_theta, t)

#index = (t_rep>-0.7) & (t_rep<0.7) & (r_rep<0.9)
#basis = basis[index]

import tables

h = tables.open_file('/mnt/stage/douwei/coeff_torch.h5')
PE_coeff = h.root.coeff[:][-1]
h.close()

'''
import h5py
with h5py.File('/mnt/stage/juno/unbinned/a/3_128.h5', "r") as ipt:
    coef = ipt['znk064'][()]['mean']
PE_coeff = np.reshape(coef[:order2,:order_eff], (-1), order='F') 
'''
tr = np.exp(np.sum(np.multiply(PE_coeff, basis), axis=1))
tr[r_rep>0.8] = np.nan
tr[t_rep>0.8] = np.nan
tr[t_rep<-0.8] = np.nan
tr = tr.reshape(-1,100)
pan = np.nansum(tr, axis=1)
#pan = np.nansum(tr[:,(t<0.7) & (t>-0.7)], axis=1)
pan[r>0.9] = np.nan
plt.figure()
plt.imshow(pan.reshape(100,100), origin='lower', extent=(-1,1,-1,1))
plt.colorbar()
plt.savefig('pan_torch.png')
plt.close()


