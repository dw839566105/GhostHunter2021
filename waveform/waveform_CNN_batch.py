import tables
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim
import pandas as pd
import pytorch_stats_loss as stats_loss
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from scipy import stats

BATCHSIZE = 30

def testing(test_loader, met='wdist') :
    batch_result = 0
    batch_count = 0
    for j, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        for batch_index_2 in range(len(outputs)):  # range(BATCHSIZE)
            # the reminder group of BATCHING may not be BATCH_SIZE
            output_vec = outputs.data[batch_index_2].cpu().numpy()
            label_vec = labels.data[batch_index_2].cpu().numpy()
            if np.sum(label_vec) <= 0:
                label_vec = np.ones(WindowSize) / 10000
            if np.sum(output_vec) <= 0:
                output_vec = np.ones(WindowSize) / 10000
            # Wdist loss
            if met == 'wdist':
                cost = stats.wasserstein_distance(np.arange(WindowSize), np.arange(WindowSize), output_vec, label_vec)
            elif met == 'l2':
                cost = np.linalg.norm(np.matmul(mnecpu, output_vec) - np.matmul(mnecpu, label_vec), ord=2)
            batch_result += cost
        batch_count += 1
    return batch_result / (BATCHSIZE * batch_count)

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
h0 = tables.open_file('/mnt/stage/douwei/GH2021/final/final-3.h5')
truth = h0.root.PETruth
EvTruth = truth[:]['EventID']
ChTruth = truth[:]['ChannelID']
TmTruth = truth[:]['PETime']
wf = h0.root.Waveform
EventID = wf[:]['EventID']
ChannelID = wf[:]['ChannelID']
Waveform = wf[:]['Waveform'] - 921
h0.close()

WindowSize = len(Waveform[0])

N1 = 200
N2 = 250
#### give PE_Truth
# H, _, _ = np.histogram2d(EvTruth,ChTruth,bins=(np.arange(2001)-0.5, np.arange(17613)-0.5))
E_bins = np.arange(N1+1) - 0.5
C_bins = np.arange(17613) - 0.5
W_bins = np.arange(1001) - 0.5

D = np.histogramdd(np.vstack((EvTruth, ChTruth, TmTruth)).T, bins=(E_bins,C_bins,W_bins))
D1 = D[0].reshape(-1,1000)

EE = np.arange(N1)
CC = np.arange(17612)
C_grid, E_grid = np.meshgrid(CC, EE, sparse=False)

left = pd.DataFrame(np.vstack((E_grid.flatten(), C_grid.flatten(), D1.T)).T)
left.columns = ['EID','CID'] + left.columns.tolist()[2:]

index = EventID < N1
right = pd.DataFrame(np.vstack((EventID[index], ChannelID[index])).T,columns=('EID','CID'))
result = pd.merge(left, right, how="right", on=["EID", "CID"])


E_bins = np.arange(N1,N2+1) - 0.5
C_bins = np.arange(17613) - 0.5
W_bins = np.arange(1001) - 0.5

D = np.histogramdd(np.vstack((EvTruth, ChTruth, TmTruth)).T, bins=(E_bins,C_bins,W_bins))
D1_test = D[0].reshape(-1,1000)

EE_test = np.arange(N1,N2)
CC = np.arange(17612)
C_grid, E_grid = np.meshgrid(CC, EE_test, sparse=False)
left_test = pd.DataFrame(np.vstack((E_grid.flatten(), C_grid.flatten(), D1_test.T)).T)
left_test.columns = ['EID','CID'] + left_test.columns.tolist()[2:]

index_test = (EventID<N2) & (EventID >= N1)
right_test = pd.DataFrame(np.vstack((EventID[index_test], ChannelID[index_test])).T,columns=('EID','CID'))
result_test = pd.merge(left_test, right_test, how="right", on=["EID", "CID"])
#### model

torch.cuda.current_device()
torch.cuda.device(0)
device = torch.device("cuda:0")

lr = 5e-3

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
x_train = torch.from_numpy(Waveform[index]).to(device).float()
y_train = torch.from_numpy(result[range(2,1002)].to_numpy()).to(device).float()

x_test = torch.from_numpy(Waveform[index_test]).to(device).float()
y_test = torch.from_numpy(result_test[range(2,1002)].to_numpy()).to(device).float()

train_dataset = Data.TensorDataset(x_train,y_train)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)

test_dataset = Data.TensorDataset(x_test,y_test)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)

testing_result = []
testing_record = open(('test_record.txt'), 'a+')

n_epoch = 300
for epoch in np.arange(n_epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        breakpoint()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = stats_loss.torch_wasserstein_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        if not (i+1) % 2000:
            print('[{0:02d}, {1:05d}] running_loss: {2:.04f}'.format(epoch, i+1, running_loss/(i+1)))
    test_performance = testing(test_loader)
    testing_record.write('{:.04f}'.format(test_performance))
    testing_result.append(test_performance)
    
    torch.save(model, 'epoch{0:02d}_loss{1:.04f}'.format(epoch, test_performance))