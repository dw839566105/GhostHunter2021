import tables
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from zernike import RZern
import h5py

path='/mnt/stage/douwei/GhostHunter'
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

def read_file_series(fileNo):
    r = np.zeros(0)
    t = np.zeros(0)
    PE = np.zeros(0)
    cos_theta = np.zeros(0)
    
    for file in np.arange(0, fileNo, 1):
        print(file)
        h = tables.open_file('%s/basis_%02d.h5' % (path, file))
        PE_tmp = h.root.PE[:]
        cos_theta_tmp = h.root.cos_theta[:]
        r_tmp = h.root.r[:]/17700
        t_tmp = h.root.t[:]
        h.close()
        r = np.hstack((r, r_tmp))
        t = t_tmp
        PE = np.hstack((PE, PE_tmp))
        cos_theta = np.hstack((cos_theta, cos_theta_tmp))
    return r, t, cos_theta, PE

def read_file_single(fileNo):
    h = tables.open_file('%s/basis_%02d.h5' % (path, file))
    PE = h.root.PE[:]
    cos_theta = h.root.cos_theta[:]
    r = h.root.r[:]/17700
    t = h.root.t[:]
    h.close()
    return r, t, cos_theta, PE

def calc_basis(order1, order2, r, cos_theta, t):
    # Zernike basis
    order1 = 2
    cart = RZern(order1)
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

    # Legendre basis
    order2 = 2
    T = legval(2*t/310-1, np.eye(order2).reshape((order2,order2,1))).T

    X_total = np.repeat(X,T.shape[0],axis=0)
    print(X_total.shape)
    T_total = np.tile(T,(X.shape[0],1))
    t_rep = np.tile(2*t/310-1, X.shape[0])
    r_rep = np.repeat(rr,T.shape[0])

    # interaction_pairs
    basis = np.zeros((X_total.shape[0], X_total.shape[1] * T_total.shape[1]))
    for index1 in np.arange(X_total.shape[1]):
        print(index1)
        for index2 in np.arange(T_total.shape[1]):
            basis[:, index2 + index1*T_total.shape[1]] = X_total[:,index1] * T_total[:,index2]
    
    return basis, t_rep, r_rep

fileNo = np.int(1)
# r, t, cos_theta, PE = read_file_single(fileNo)
r, t, cos_theta, PE = read_file_series(fileNo)
basis, t_rep, r_rep = calc_basis(10, 10, r, cos_theta, t)
index = (t_rep>-0.7) & (t_rep<0.7) & (r_rep<0.9)
basis = basis[index]
PE = PE[index]

from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
dtype = torch.float
device = torch.device("cpu")

y = torch.from_numpy(PE).to(device).float()
coeff_new = torch.zeros(basis.shape[1]).to(device).float()
coeff_new[0] = - 0.7
for iteration in np.arange(1, basis.shape[1]):
    x = torch.from_numpy(basis[:,:iteration]).to(device).float()
    basis = torch.from_numpy(basis[:,:iteration]).to(device).float()
    # coeff_new[0] = torch.log(torch.mean(y))
    class DynamicNet(torch.nn.Module):
        def __init__(self, coeff_var, basis_var):
            """
            In the constructor we instantiate five parameters and assign them as members.
            """
            super().__init__()
            # Randomly initialize weights
            # self.coeff = torch.nn.Parameter(
            #   torch.from_numpy(coeff).to(device).float())
            self.coeff = torch.nn.Parameter(coeff_var)
            # self.coeff = torch.nn.Parameter(
            #     torch.randn(basis.shape[1],1))
        def forward(self, x):
            y = torch.matmul(x, self.coeff)
            return y
    
    # device = torch.device("cuda:0") # Uncomment this to run on GPU
    
    dataset = TensorDataset(x, y)
    #train_dataset, val_dataset = random_split(dataset, [np.int(y.shape[0]*0.8), np.int(y.shape[0]*0.2)])

    #train_loader = DataLoader(dataset=dataset, batch_size=np.int(y.shape[0]/fileNo/5 + 1))
    train_loader = DataLoader(dataset=dataset, batch_size=np.int(y.shape[0]))
    #val_loader = DataLoader(dataset=val_dataset, batch_size=20)
    print('='*80)
    
    model = DynamicNet(coeff_new[:iteration], basis)
    criterion = torch.nn.PoissonNLLLoss(log_input=True,reduction='mean')
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    #optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.01), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    n_epochs = 100
    losses = []
    coeff_save = np.zeros((basis.shape[1],0))

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
            with h5py.File('coeff_torch_step.h5', 'w') as out:
                out.create_dataset('coeff%d' % iteration, data = coeff_save)
        scheduler.step()
        print(scheduler.get_last_lr())
    print('epoch:%d' % epoch)
    coeff_new = model.coeff
