import tables
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from zernike import RZern

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

r = np.zeros(0)
t = np.zeros(0)
PE = np.zeros(0)
cos_theta = np.zeros(0)

N_max = np.int(1e6)
for file in np.arange(0, 11, 1):
    print(file)
    h = tables.open_file('basis_%02d_small.h5' % file)
    PE_tmp = h.root.PE[:]
    cos_theta_tmp = h.root.cos_theta[:]
    r_tmp = h.root.r[:]/17700
    t_tmp = h.root.t[:]
    h.close()
    r = np.hstack((r, r_tmp))
    t = t_tmp
    PE = np.hstack((PE, PE_tmp))
    cos_theta = np.hstack((cos_theta, cos_theta_tmp))

order = 20
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

order2 = 20

T = legval(2*t/310-1, np.eye(order2).reshape((order2,order2,1))).T

X_total = np.repeat(X,T.shape[0],axis=0)
T_total = np.tile(T,(X.shape[0],1))
import pandas as pd
L_strings = ["L%d" % x for x in np.arange(X_total.shape[1])]
L_pd = pd.DataFrame(X_total, columns=L_strings)

t_strings = ["T%d" % x for x in np.arange(T_total.shape[1])]
t_pd = pd.DataFrame(T_total, columns=t_strings)

output = pd.DataFrame(PE, columns=['Y'])

import pyarrow as pa
import pyarrow.parquet as pq
frames = [L_pd, t_pd, output]
data = pd.concat(frames, axis=1)
table = pa.Table.from_pandas(data)
pq.write_table(table, 'Juno2_20_20.parquet')
