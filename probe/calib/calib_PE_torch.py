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
        h = tables.open_file('/home/douwei/JUNO_probe/basis_%02d_PE.h5' % file)
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

fileNo = np.int32(3)
# r, t, cos_theta, PE = read_file_single(fileNo)
r, cos_theta, PE = read_file_series(fileNo)
X = calc_basis(r, cos_theta, order=order)

# model = sm.GLM(np.atleast_2d(PE).T, X, family=sm.families.Poisson())
# res = model.fit()
# print(res.summary())
# coef_ = res.params
# with h5py.File('coeff_PE.h5', 'w') as out:
#    out.create_dataset('coeff', data = coef_)
# plot(coef_, order=order)

from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
dtype = torch.float
device = torch.device("cpu")

x = torch.from_numpy(X).to(device).float()
y = torch.from_numpy(PE).to(device).float()

coeff_new = torch.zeros(x.shape[1]).float()
coeff_new[0] = torch.log(torch.mean(y))

class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        # Randomly initialize weights
        # self.coeff = torch.nn.Parameter(
        #   torch.from_numpy(coeff).to(device).float())
        self.coeff = torch.nn.Parameter(coeff_new)
        # self.coeff = torch.nn.Parameter(
        #     torch.randn(basis.shape[1],1))
    def forward(self, x):
        y = torch.matmul(x, self.coeff)
        return y
    
dataset = TensorDataset(x, y)
#train_dataset, val_dataset = random_split(dataset, [np.int(y.shape[0]*0.8), np.int(y.shape[0]*0.2)])

train_loader = DataLoader(dataset=dataset, batch_size=np.int(y.shape[0]/fileNo/5 + 1))
#val_loader = DataLoader(dataset=val_dataset, batch_size=20)
print('='*80)
model = DynamicNet()
criterion = torch.nn.PoissonNLLLoss(log_input=True,reduction='mean')
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
# scheduler = ExponentialLR(optimizer, gamma=0.5)
n_epochs = 1000
losses = []

record = []
for epoch in range(n_epochs):
    # Forward pass: compute predicted y
    for index, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        print(epoch, index, loss.item())
        print(model.coeff[0:5].t())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

        tmp = model.coeff
        record.append(tmp.detach().numpy())
        coeff_save = np.array(record)
    scheduler.step()
    print(scheduler.get_last_lr())