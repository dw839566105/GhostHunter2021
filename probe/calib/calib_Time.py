import tables
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from zernike import RZern
import sys
from numba import jit
import h5py

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

order = np.int32(eval(sys.argv[1]))

N_max = np.int64(1e6)

def read_file_series(fileNo):
    PE = np.zeros(0)
    T = np.zeros(0)
    
    for file in np.arange(0, fileNo, 1):
        print(file)
        h = tables.open_file('/mnt/stage/douwei/GhostHunter/basis_%02d_Time.h5' % file)
        PE_tmp = h.root.PE[:]
        T_tmp = h.root.Time[:]
        h.close()

        PE = np.hstack((PE, PE_tmp))
        T = np.hstack((T, T_tmp))
    return PE, T
    
def calc_basis(T, order=10):
    T = 2*(T-160)/320
    X = legval(T, np.eye(order).reshape((order, order, 1))).T
    return X

def plot(coef_, order):
    L, K = 500, 500
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart = RZern(order)
    cart.make_cart_grid(xv, yv)
    nk = cart.nk
    m = cart.mtab
    n = cart.ntab
    coef = np.zeros(nk)
    coef[m>=0] = coef_
    # normal scale
    # im = plt.imshow(np.exp(cart.eval_grid(np.array(coef_), matrix=True)), origin='lower', extent=(-1, 1, -1, 1))
    # log scale
    im = plt.imshow(cart.eval_grid(np.array(coef), matrix=True), origin='lower', extent=(-1, 1, -1, 1))
    plt.colorbar(im)
    plt.savefig('Legendre_%d_log.png' % order)
    plt.close()
    plt.figure()
    im = plt.imshow(np.exp(cart.eval_grid(np.array(coef), matrix=True)), origin='lower', extent=(-1, 1, -1, 1))
    plt.colorbar(im)
    plt.savefig('Legendre_%d.png' % order)
    plt.close()

    
import statsmodels.api as sm

fileNo = np.int32(20)
# r, t, cos_theta, PE = read_file_single(fileNo)
PE, T = read_file_series(fileNo)
X = calc_basis(T, order=order)
model = sm.GLM(np.atleast_2d(PE).T, X, family=sm.families.Poisson())
res = model.fit()
print(res.summary())
coef_ = res.params
with h5py.File('coeff_Time.h5', 'w') as out:
    out.create_dataset('coeff', data = coef_)
#plot(coef_, order=order)
