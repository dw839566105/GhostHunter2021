import tables
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim
import pandas as pd
import pytorch_stats_loss as stats_loss
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 33, 33, padding=16)
        self.conv2 = nn.Conv1d(33, 17, 17, padding=8)
        self.conv3 = nn.Conv1d(17, 13, 13, padding=6)
        self.conv4 = nn.Conv1d(13, 9, 9, padding=4)
        self.conv5 = nn.Conv1d(9, 5, 5, padding=2)
        self.conv6 = nn.Conv1d(5, 1, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.05)
        drop_out = nn.Dropout(0.9)
        x = torch.unsqueeze(x, 1)
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))
        x = leaky_relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.squeeze(1)
        return x
#### load data
h0 = tables.open_file('/mnt/stage/douwei/GH2021/final/final-2.h5')
truth = h0.root.PETruth
EvTruth = truth[:]['EventID']
ChTruth = truth[:]['ChannelID']
TmTruth = truth[:]['PETime']
wf = h0.root.Waveform
EventID = wf[:]['EventID']
ChannelID = wf[:]['ChannelID']
Waveform = wf[:]['Waveform']
h0.close()

N = 200
#### give PE_Truth
# H, _, _ = np.histogram2d(EvTruth,ChTruth,bins=(np.arange(2001)-0.5, np.arange(17613)-0.5))
E_bins = np.arange(N+1) - 0.5
C_bins = np.arange(17613) - 0.5
W_bins = np.arange(1001) - 0.5

D = np.histogramdd(np.vstack((EvTruth, ChTruth, TmTruth)).T, bins=(E_bins,C_bins,W_bins))
D1 = D[0].reshape(-1,1000)

EE = np.arange(N)
CC = np.arange(17612)
C_grid, E_grid = np.meshgrid(CC, EE, sparse=False)

left = pd.DataFrame(np.vstack((E_grid.flatten(), C_grid.flatten(), D1.T)).T)

left.columns = ['EID','CID'] + left.columns.tolist()[2:]

'''
E_grid, C_grid = np.meshgrid(EE, CC, sparse=False)
import pandas as pd
left = pd.DataFrame(np.vstack((E_grid.flatten(), C_grid.flatten(), H.T.flatten())).T, 
                    columns=('EID','CID','Q'))

left1 = pd.DataFrame(np.vstack((E_grid.flatten(), C_grid.flatten(), H.T.flatten())).T, 
                    columns=('EID','CID','PETime'))
'''

index = EventID<N
right = pd.DataFrame(np.vstack((EventID[index], ChannelID[index])).T,columns=('EID','CID'))
result = pd.merge(left, right, how="right", on=["EID", "CID"])

#### model

torch.cuda.current_device()
torch.cuda.device(0)
device = torch.device("cuda:0")

lr = 5e-3

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
x_train = torch.from_numpy(Waveform[index]).to(device).float()
y_train = torch.from_numpy(result[range(2,1002)].to_numpy()).to(device).float()

breakpoint()
n_epoch = 300
for epoch in np.arange(n_epoch):
    running_loss = 0.0
    for batch in np.arange(x_train.shape[0]):
        pred = model(x_train[batch:batch+1])
        optimizer.zero_grad()
        loss = stats_loss.torch_wasserstein_loss(pred, y_train[batch:batch+1])
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        if not batch % 2000:
            print('[{0:02d}, {1:05d}] running_loss: {2:.04f}'.format(epoch, batch, running_loss/batch))
    torch.save(model, 'haha')




