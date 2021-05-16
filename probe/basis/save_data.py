import tables
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

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
for radius in np.arange(500, 18000, 500):
    print(radius)
    h = tables.open_file('basis_split_%d.h5' % radius)
    PE_tmp = h.root.PE[:]
    cos_theta_tmp = h.root.cos_theta[:]
    r_tmp = h.root.r[:]
    t_tmp = h.root.t[:]
    h.close()
    if(PE_tmp.shape[0]>N_max):
        PE_tmp = PE_tmp[:N_max]
        cos_theta_tmp = cos_theta_tmp[:N_max]
        r_tmp = r_tmp[:N_max]
        t_tmp = t_tmp[:N_max]

    r = np.hstack((r, r_tmp))
    t = np.hstack((t, t_tmp))
    PE = np.hstack((PE, PE_tmp))
    cos_theta = np.hstack((cos_theta, cos_theta_tmp))
    
cut1 = 20
cut2 = 5

xx = legval(cos_theta, np.eye(cut1).reshape((cut1,cut1,1))).T
rr = legval(2*r/np.max(r)-1, np.eye(cut1).reshape((cut1,cut1,1))).T
tt = legval(2*t/np.max(t)-1, np.eye(cut2).reshape((cut2,cut2,1))).T

import pandas as pd
x_strings = ["x%d" % x for x in np.arange(cut1)]
x_pd = pd.DataFrame(xx, columns=x_strings)

r_strings = ["r%d" % x for x in np.arange(cut1)]
r_pd = pd.DataFrame(rr, columns=r_strings)

LL = np.zeros((xx.shape[0], cut1*cut1))
L_strings = []
for i in np.arange(cut1):
    print(i)
    for j in np.arange(cut1):
        LL[:, i*cut1+j] = xx[:, i]*rr[:, j]
        L_strings.append("L%d" % (i*cut1+j))
L_pd = pd.DataFrame(LL, columns=L_strings)        
#L_strings = ["L%d" % x for x in np.arange(cut1**2)]

t_strings = ["t%d" % x for x in np.arange(cut2)]
t_pd = pd.DataFrame(tt, columns=t_strings)

output = pd.DataFrame(PE, columns=['Y'])

import pyarrow as pa
import pyarrow.parquet as pq
frames = [L_pd, t_pd, output]
data = pd.concat(frames, axis=1)
table = pa.Table.from_pandas(data)
pq.write_table(table, 'test1.parquet')
