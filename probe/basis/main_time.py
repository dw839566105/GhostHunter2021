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

PE = np.zeros((init_x.shape[0], time_bins[1:].shape[0]))

Time = np.tile(time_bins[1:],(10,1))

vertex = np.vstack((init_x, init_y, init_z)).T
for j_index, j in enumerate(np.unique(EventId)):
    print(j)
    Time_single, _ = np.histogram(HitTime[EventId==j], bins=time_bins)
    PE[j_index] = Time_single

with h5py.File(output, 'w') as out:
    out.create_dataset('PE', data = PE.flatten())
    out.create_dataset('Time', data = Time.flatten())
