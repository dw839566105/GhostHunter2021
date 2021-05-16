import numpy as np
import tables
import matplotlib.pyplot as plt
import h5py
import sys

file = sys.argv[1]
output = sys.argv[2]

path = '/junofs/users/junoprotondecay/zhangaq/MIX/simulation/e-/uniform/'

PMT_pos = np.loadtxt('/junofs/users/junoprotondecay/xubd/harvest/data/geo.csv')
x = np.sin(PMT_pos[:,1]/180*np.pi) * np.cos(PMT_pos[:,2]/180*np.pi)
y = np.sin(PMT_pos[:,1]/180*np.pi) * np.sin(PMT_pos[:,2]/180*np.pi)
z = np.cos(PMT_pos[:,1]/180*np.pi)
PMT_all = np.vstack((x, y, z)).T
PMTId = PMT_pos[:,0]
bound = (np.max(PMTId[PMTId<100000])+2).astype('int')
PMT = PMT_all[:bound-1]
PMTNo = PMT.shape[0]

PMT_bins = np.arange(0,bound) - 0.5
#time_bins = np.arange(1, 1024, 5)
time_bins = np.arange(10, 310, 2)
h = tables.open_file(path + '%s.h5' % file)
EventId = h.root.SimEvent.SimCDHit[:]['event_id']
PMTId = h.root.SimEvent.SimCDHit[:]['pmt_id']
HitTime = h.root.SimEvent.SimCDHit[:]['hit_time']
init_x = h.root.SimEvent.SimTrack[:]['init_x']
init_y = h.root.SimEvent.SimTrack[:]['init_y']
init_z = h.root.SimEvent.SimTrack[:]['init_z']
h.close()

print(init_x.shape[0])
EventNo = init_x.shape[0]
mesh = np.meshgrid(PMT_bins[1:], time_bins[1:], sparse=False)
#print(PMT.shape)
#print(PMT_bins.shape)
PE = np.zeros((init_x.shape[0], mesh[0].flatten().shape[0]))

cos_theta = np.zeros((init_x.shape[0], PMTNo))

vertex = np.vstack((init_x, init_y, init_z)).T
for j_index, j in enumerate(np.unique(EventId)):
    print(j)
    PE_single, xedges, yedges = np.histogram2d(PMTId[EventId==j], HitTime[EventId==j], 
                                               bins=(PMT_bins, time_bins))
    PE[j_index][:] = PE_single.flatten()
    
    vertex_single = np.repeat(np.atleast_2d(vertex[j_index]), PMTNo, axis=0)/1e3
    cos_theta[j_index] = np.sum(vertex_single*PMT, axis=1)/ \
            np.linalg.norm(vertex_single, axis=1)/np.linalg.norm(PMT, axis=1)
'''
for j_index, j in enumerate(np.unique(EventId)):
    print(j)
    PE_single, xedges, yedges = np.histogram2d(PMTId[EventId==j], HitTime[EventId==j], 
                                               bins=(PMT_bins, time_bins))
    PE[j_index][:] = PE_single.flatten()
    
    vertex_single = np.repeat(np.atleast_2d(vertex[j_index]), PMTNo, axis=0)/1e3
    cos_theta[j_index] = np.tile(np.sum(vertex_single*PMT, axis=1)/ \
            np.linalg.norm(vertex_single, axis=1)/np.linalg.norm(PMT, axis=1), len(time_bins[:-1]))
    
base_r, base_theta, base_t = np.meshgrid(np.linalg.norm(vertex, axis=1), PMT_bins[:-1], time_bins[:-1],  
                                         sparse=False)
r = base_r.flatten()
theta = base_theta.flatten()
t = base_t.flatten()
'''

r = np.linalg.norm(vertex, axis=1)
theta = PMT_bins[:-1]
t = time_bins[:-1]
print(PE.shape, cos_theta.shape, r.shape, theta.shape, t.shape)
with h5py.File(output, 'w') as out:
    out.create_dataset('PE', data = PE.flatten())
    out.create_dataset('cos_theta', data = cos_theta.flatten())
    out.create_dataset('r', data = r)
    out.create_dataset('theta', data = theta)
    out.create_dataset('t', data = t)
