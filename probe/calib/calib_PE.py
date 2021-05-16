import tables
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from zernike import RZern
import h5py
import sys

order = np.int32(eval(sys.argv[1]))
r = np.zeros(0)
t = np.zeros(0)
PE = np.zeros(0)
cos_theta = np.zeros(0)

N_max = np.int64(1e6)

def read_file_series(fileNo):
    r = np.zeros(0)
    PE = np.zeros(0)
    cos_theta = np.zeros(0)
    
    for file in np.arange(0, fileNo, 1):
        print(file)
        h = tables.open_file('/mnt/stage/douwei/GhostHunter/basis_%02d_PE.h5' % file)
        PE_tmp = h.root.PE[:]
        cos_theta_tmp = h.root.cos_theta[:]
        r_tmp = h.root.r[:]/17700
        h.close()
        r = np.hstack((r, r_tmp))
        PE = np.hstack((PE, PE_tmp))
        cos_theta = np.hstack((cos_theta, cos_theta_tmp))
    return r, cos_theta, PE
    
def calc_basis(r, cos_theta, order=10):
    # Zernike basis
    cart = RZern(order)
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
    plt.savefig('Zernike_%d_log.png' % order)
    plt.close()
    plt.figure()
    im = plt.imshow(np.exp(cart.eval_grid(np.array(coef), matrix=True)), origin='lower', extent=(-1, 1, -1, 1))
    plt.colorbar(im)
    plt.savefig('Zernike_%d.png' % order)
    plt.close()

    
import statsmodels.api as sm

fileNo = np.int32(20)
# r, t, cos_theta, PE = read_file_single(fileNo)
r, cos_theta, PE = read_file_series(fileNo)
X = calc_basis(r, cos_theta, order=order)
model = sm.GLM(np.atleast_2d(PE).T, X, family=sm.families.Poisson())
res = model.fit()
print(res.summary())
coef_ = res.params
with h5py.File('coeff_PE.h5', 'w') as out:
    out.create_dataset('coeff', data = coef_)
# plot(coef_, order=order)


