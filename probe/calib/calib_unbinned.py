import tables
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")

h = tables.open_file('../basis/Event10_offset.h5')
ohitt = h.root.hitt[:]
ohitz = h.root.hitz[:]
ononhitt = h.root.nonhitt[:]
ononhitz = h.root.nonhitz[:]
ooffset = h.root.offset[:]

h.close()
hitt = torch.from_numpy(ohitt).to(device).float()
hitz = torch.from_numpy(ohitz).to(device).float()
nonhitt = torch.from_numpy(ononhitt).to(device).float()
nonhitz = torch.from_numpy(ononhitz).to(device).float()
offset = torch.from_numpy(ooffset + 1.022).to(device).float()

n_epochs = 10000

coeff_ini = torch.zeros((ohitt.shape[0], ohitt.shape[0])).float()
coeff_last = torch.ones(0,0).float()

Z_order = 1
for T_order in range(1, ohitt.shape[0]+1): 
    # coeff_flatten = torch.flatten(coeff_ini)
    coeff = torch.zeros((Z_order, T_order))
    if(Z_order + T_order>2):
        coeff[:,:-1] = coeff_last
    else:
        coeff = -20*torch.ones(1,1).float()
    nonhitt_tmp = nonhitt[:T_order,:]
    nonhitz_tmp = nonhitz[:Z_order,:]
    hitt_tmp = hitt[:T_order,:]
    hitz_tmp = hitz[:Z_order,:]

    class DynamicNet(torch.nn.Module):
        def __init__(self, coeff):
            """
            In the constructor we instantiate five parameters and assign them as members.
            """
            super().__init__()
            self.coeff = torch.nn.Parameter(coeff)

        def forward(self):
            #breakpoint()
            #coeff_test = -17*torch.nn.Parameter(torch.ones((1,1)))
            #torch.sum(nonhitt_tmp.T @ coeff_test @ nonhitz_tmp) - torch.sum(torch.sum((hitt_tmp @ hitz_tmp.T) * coeff_test) / nonhitz_tmp.shape[1])
            nonhit = torch.exp(nonhitt_tmp.T @ self.coeff.T @ nonhitz_tmp)*offset
            #nonhit = torch.exp(nonhitt_tmp.T @ self.coeff.T @ nonhitz_tmp)
            hit = torch.sum((hitt_tmp @ hitz_tmp.T) * self.coeff) / nonhitz_tmp.shape[1]
            return -(torch.sum(hit) - torch.sum(nonhit))

    print('='*80)
    model = DynamicNet(coeff)
    #criterion = torch.nn.MSELoss(log_input=True, reduction='mean')
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    # scheduler = ExponentialLR(optimizer, gamma=0.5)

    for epoch in range(n_epochs):
        # Forward pass: compute predicted y
        y_pred = model()
        #loss = criterion(y_pred, y_batch)
        if not epoch % 10:
            print(epoch, y_pred.item(), model.coeff)
        if(epoch>0):    
            if(torch.abs(old-y_pred)<1e-4):
                break
        optimizer.zero_grad()
        y_pred.backward()
        optimizer.step()
        old = y_pred.item()
    coeff_last = model.coeff
    torch.save(model.coeff, 'coeff_Z%d_T%d' % (Z_order, T_order))

for Z_order in range(2, ohitz.shape[0]+1): 
    # coeff_flatten = torch.flatten(coeff_ini)
    coeff = torch.zeros((Z_order, T_order))
    coeff[:-1,:] = coeff_last
    nonhitt_tmp = nonhitt[:T_order,:]
    nonhitz_tmp = nonhitz[:Z_order,:]
    hitt_tmp = hitt[:T_order,:]
    hitz_tmp = hitz[:Z_order,:]

    class DynamicNet(torch.nn.Module):
        def __init__(self, coeff):
            """
            In the constructor we instantiate five parameters and assign them as members.
            """
            super().__init__()
            self.coeff = torch.nn.Parameter(coeff)

        def forward(self):
            #breakpoint()
            #coeff_test = -17*torch.nn.Parameter(torch.ones((1,1)))
            #torch.sum(nonhitt_tmp.T @ coeff_test @ nonhitz_tmp) - torch.sum(torch.sum((hitt_tmp @ hitz_tmp.T) * coeff_test) / nonhitz_tmp.shape[1])
            nonhit = torch.exp(nonhitt_tmp.T @ self.coeff.T @ nonhitz_tmp)*offset
            #nonhit = torch.exp(nonhitt_tmp.T @ self.coeff.T @ nonhitz_tmp)
            hit = torch.sum((hitt_tmp @ hitz_tmp.T) * self.coeff) / nonhitz_tmp.shape[1]
            return -(torch.sum(hit) - torch.sum(nonhit))

    print('='*80)
    model = DynamicNet(coeff)
    #criterion = torch.nn.MSELoss(log_input=True, reduction='mean')
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    # scheduler = ExponentialLR(optimizer, gamma=0.5)

    for epoch in range(n_epochs):
        # Forward pass: compute predicted y
        y_pred = model()
        #loss = criterion(y_pred, y_batch)
        if not epoch % 10:
            print(epoch, y_pred.item(), model.coeff)
        if(epoch>0):    
            if(torch.abs(old-y_pred)<1e-4):
                break
        optimizer.zero_grad()
        y_pred.backward()
        optimizer.step()
        old = y_pred.item()
    coeff_last = model.coeff
    torch.save(model.coeff, 'coeff_Z%d_T%d' % (Z_order, T_order))