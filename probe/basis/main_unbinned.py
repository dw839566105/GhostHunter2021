import numpy as np
import tables
import matplotlib.pyplot as plt
import h5py
import sys

path = sys.argv[1]
output = sys.argv[2]

from numba import jit
from zernike import RZern
@jit
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

@jit
def angular(m, theta):
    return np.cos(m * theta)

@jit
def polyval(p, x):
    y = np.zeros((p.shape[1], x.shape[0]))
    for i in range(len(p)):
        y = y * x + p[i]
    return y

def ReadJunoPMT(file = r'/junofs/users/junoprotondecay/xubd/harvest/data/geo.csv'):
    A = np.loadtxt(file)
    x = 17.7 * np.sin(A[:,1]/180*np.pi) * np.cos(A[:,2]/180*np.pi)
    y = 17.7 * np.sin(A[:,1]/180*np.pi) * np.sin(A[:,2]/180*np.pi)
    z = 17.7 * np.cos(A[:,1]/180*np.pi)
    
    '''
    Gdata = np.loadtxt('/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/data/Simulation/ElecSim/pmtdata.txt',dtype=bytes).astype('str')
    G = np.setdiff1d(Gdata[:,0].astype('int'),A[:,0])

    GG = Gdata[:,0].astype('int')
    id1 = np.setdiff1d(GG,A[:,0])

    Gtype = Gdata[GG!=id1,1]
    GGain = Gdata[GG!=id1,2].astype('float')
    '''
    PMT_pos = (np.vstack((x,y,z)).T)[A[:,0]<100000]

    return PMT_pos, A[A[:,0]<100000, 0]

def ReadFile(filename):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h = tables.open_file(filename,'r')
    print(filename, flush=True)
    ParticleTruth = h.root.ParticleTruth
    EventID = ParticleTruth[:]['EventID']
    photon_vis = ParticleTruth[:]['vis_photons']
    x = ParticleTruth[:]['x']/1000
    y = ParticleTruth[:]['y']/1000
    z = ParticleTruth[:]['z']/1000
    E_vis = ParticleTruth[:]['p']
    
    PETruth = h.root.PETruth
    EId = PETruth[:]['EventID']
    Ch = PETruth[:]['ChannelID']
    time = PETruth[:]['PETime']

    h.close()
    return EventID, x, y, z, E_vis,\
        EId, Ch, time

def ReadChain(path):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    EventID = []
    for series in np.arange(2,3):
        # filename = path + '1t_' + radius + '.h5'
        # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
        filename = '%s/final-%d.h5' % (path, series)

        if len(EventID) == 0:
            EventID, x, y, z, E_vis, EId, Ch, time = ReadFile(filename)
        else:
            EventID_, x_, y_, z_, E_vis_, EId_, Ch_, time_ = ReadFile(filename)
            if(len(EventID_)>0):
                EventID_ += np.max(EventID) + 1
                EId_ += np.max(EventID) + 1
                EventID = np.hstack((EventID, EventID_))
                x = np.hstack((x, x_))
                y = np.hstack((y, y_))
                z = np.hstack((z, z_))
                E_vis = np.hstack((E_vis, E_vis_))
                EId = np.hstack((EId, EId_))
                Ch = np.hstack((Ch, Ch_))
                time = np.hstack((time, time_))
        print('total event: ', len(EventID))
    return EventID, x, y, z, E_vis, EId, Ch, time

def Zernike(r, theta, order):
    cart = RZern(order)
    zo = cart.mtab >= 0
    zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], r)
    zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), theta)
    nonhitz = zs_radial * zs_angulars
    return nonhitz

EventID, x, y, z, E_vis, EId, Ch, time = ReadChain(path)

PMT,_ = ReadJunoPMT()
###### for non-hit
vertex = np.vstack((x,y,z)).T/17.7
r = np.linalg.norm(vertex, axis=1)

v_rep = np.repeat(vertex, PMT.shape[0], axis=0)
r_rep = np.repeat(r, PMT.shape[0], axis=0)
p_rep = np.tile(PMT, (vertex.shape[0],1))
cos_theta = np.sum(v_rep*p_rep, axis=1)/np.linalg.norm(v_rep, axis=1)/np.linalg.norm(p_rep, axis=1)
theta = np.arccos(cos_theta)

N = 100
t = np.linspace(-1,1,N)

order_T = 40
order_Z = 100
coeff = np.eye(order_T).reshape((order_T, order_T, 1))

# non-hit time: event * time order
nonhitt = legval(t, coeff)
# non-hit z: event * zernike order
nonhitz = Zernike(r_rep, theta, order_Z)

###### for hit
# transform to [-1,1]
t_min = -40
t_max = 1000

# hit t:
time_leg = (time - t_min)/(t_max - t_min)*2 - 1
hitt = legval(time_leg, coeff)
# hit z
counts = np.bincount(EId)
counts = counts[counts!=0]
p_rep_ = PMT[Ch]
v_rep_ = np.repeat(vertex, counts, axis=0)
r_rep_ = np.repeat(r, counts, axis=0)
cos_theta_ = np.sum(v_rep_*p_rep_, axis=1)/np.linalg.norm(v_rep_, axis=1)/np.linalg.norm(p_rep_, axis=1)
theta_ = np.arccos(cos_theta_)
hitz = Zernike(r_rep_, theta_, order_Z)

breakpoint()

with h5py.File(output, 'w') as out:
    out.create_dataset('hitt', data = hitt)
    out.create_dataset('hitz', data = hitz)
    out.create_dataset('nonhitt', data = nonhitt)
    out.create_dataset('nonhitz', data = nonhitz)