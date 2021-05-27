# recon range: [-1,1], need * detector radius
import numpy as np
import tables
import os,sys
import argparse
from scipy.optimize import minimize

from numba import jit
from zernike import RZern
import warnings
warnings.filterwarnings('ignore')
sys.stdout.flush()
np.set_printoptions(precision=3, suppress=True)

class ReconData(tables.IsDescription):
    EventID = tables.Int64Col(pos=0)    # EventNo
    # inner recon
    E = tables.Float16Col(pos=1)        # Energy
    x = tables.Float16Col(pos=2)        # x position
    y = tables.Float16Col(pos=3)        # y position
    z = tables.Float16Col(pos=4)        # z position
    t0 = tables.Float16Col(pos=5)          # time offset
    success = tables.Int64Col(pos=6)        # recon status   
    Likelihood = tables.Float16Col(pos=7)
    
# boundaries
shell = 17.7

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

@jit
def angular(m, theta):
    return np.cos(m * theta)

@jit
def polyval(p, x):
    y = np.zeros((p.shape[1], x.shape[0]))
    for i in range(len(p)):
        y = y * x + p[i]
    return y

def load_coeff():

    # PE Zernike coefficients
    h = tables.open_file('PE_dns_15.h5','r')
    coeff_pe = h.root.coeff15[:]
    h.close()

    # time Zernike coefficients
    h = tables.open_file('Time_vis_dn0_20.h5','r')
    coeff_time = h.root.coeff20[:]
    h.close()
    return coeff_pe, coeff_time

def r2c(c):
    v = np.zeros(3)
    v[2] = c[0] * np.cos(c[1]) #z
    rho = c[0] * np.sin(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def c2r(c):
    v = np.zeros(3)
    v[0] = np.linalg.norm(c)
    v[1] = np.arccos(c[2]/(v[0]+1e-6))
    #v[2] = np.arctan(c[1]/(c[0]+1e-6)) + (c[0]<0)*np.pi
    v[2] = np.arctan2(c[1],c[0])
    return v

def ReadJunoPMT(file = r'/junofs/users/junoprotondecay/xubd/harvest/data/geo.csv'):
    A = np.loadtxt(file)
    x = 17.7 * np.sin(A[:,1]/180*np.pi) * np.cos(A[:,2]/180*np.pi)
    y = 17.7 * np.sin(A[:,1]/180*np.pi) * np.sin(A[:,2]/180*np.pi)
    z = 17.7 * np.cos(A[:,1]/180*np.pi)
    
    PMT_pos = (np.vstack((x,y,z)).T)[A[:,0]<100000]
    return PMT_pos, A[A[:,0]<100000, 0]

def Likelihood(vertex, *args):
    '''
    vertex[1]: r
    vertex[2]: theta
    vertex[3]: phi
    '''
    PMT_pos, fired_PMT, time_array, pe_array, weight_array= args
    basis = Calc_basis(vertex, PMT_pos)
    L1 = Likelihood_PE(basis, pe_array)
    L2 = Likelihood_Time(basis, vertex[-1], fired_PMT, time_array, weight_array)
    return L1 + L2

def Calc_basis(vertex, PMT_pos): 
    # boundary
    v = r2c(vertex[:3])
    rho = vertex[0]
    if rho > 1-1e-3:
        rho = 1-1e-3
    # calculate cos theta
    cos_theta = np.dot(v, PMT_pos.T) / (np.linalg.norm(v)*np.linalg.norm(PMT_pos,axis=1))   
    cos_theta = np.nan_to_num(cos_theta)
    theta = np.arccos(cos_theta)
    # Generate Zernike basis
    
    basis = np.zeros((len(PMT_pos), nk))
    '''traditional
    for i in np.arange(nk):
        if m[i] >= 0:
            basis[:,i] = cart.Zk(i, rho, np.arccos(cos_theta))
    '''
    zo = cart.mtab >= 0
    rho = rho + np.zeros_like(seq)
    theta = theta + np.zeros_like(seq)
    zs_radial = cart.coefnorm[zo, np.newaxis] * polyval(cart.rhotab.T[:, zo, np.newaxis], rho)
    zs_angulars = angular(cart.mtab[zo].reshape(-1, 1), theta)
    basis_pos = zs_radial * zs_angulars
    basis[:,zo] = basis_pos.T
    return basis
    
def Likelihood_PE(basis, pe_array):
    expect = np.exp(np.matmul(basis[:,n<=15], coef_PE))

    # Energy fit 
    nml = np.sum(expect)/np.sum(pe_array)
    expect = expect/nml

    # Poisson likelihood
    # p(q|lambda) = sum_n p(q|n)p(n|lambda)
    #         = sum_n Gaussian(q, n, sigma_n) * exp(-expect) * expect^n / n!
    # int p(q|lambda) dq = sum_n exp(-expect) * expect^n / n! = 1
    a0 = expect ** pe_array
    a2 = np.exp(-expect)

    # -ln Likelihood
    L = - np.sum(np.log(a0*a2))
    return L

def Likelihood_Time(basis, T0, fired_PMT, time_array, weight_array):
    basis_time = basis[fired_PMT]
    
    # Recover coefficient
    T_i = np.matmul(basis_time, coef_time)
    T_i = T_i + T0
    
    # Likelihood
    L = - np.nansum(Likelihood_quantile(time_array, weight_array, T_i, 0.12, 3))
    return L

def Likelihood_quantile(y, weight_array, T_i, tau, ts):
    # less = T_i[y<T_i] - y[y<T_i]
    # more = y[y>=T_i] - T_i[y>=T_i]    
    # R = (1-tau)*np.sum(less) + tau*np.sum(more)
    
    # since lucy ddm is not sparse, use PE as weight
    R = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
    nml = (tau*(1-tau)/ts)**weight_array
    L0 = -1/ts * R * weight_array + np.log(nml)
    
    #nml = tau*(1-tau)/ts
    #L_norm = np.exp(-np.atleast_2d(L).T) * nml / ts
    #L = np.sum(np.log(L_norm), axis=1)
    #L0 = L/ts
    return L0

def recon(fin, fout):

    '''
    reconstruction

    fid: root reference file convert to .h5
    fout: output file
    '''
    # Create the output file and the group
    print(fin) # filename

    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    # Loop for event

    h_PE = tables.open_file('/mnt/stage/douwei/GH2021/final-data/charge/wf_charge_%03d.h5' % eval(fin),'r')
    charge = h_PE.root.charge[:]
    EventID = h_PE.root.EventID[:]
    ChannelID = h_PE.root.ChannelID[:]
    h_PE.close()

    h_time = tables.open_file('/mnt/stage/douwei/GH2021/final-data/time/wf_time_%03d.h5' % eval(fin),'r')
    time = h_time.root.time[:]
    h_time.close()
   
    _, PMT_No = ReadJunoPMT()
    charge/=-160
    charge[charge<0] = 0

    time = time/np.sum(time,axis=1)[:,np.newaxis]*charge[:,np.newaxis]
    #charge = np.round(charge)
    
    
    time_series, ch_series = np.meshgrid(np.arange(time.shape[1]), np.arange(time.shape[0]))
    weight_array = time.flatten()
    time_array = time_series.flatten()
    fired_PMT = ch_series.flatten()
    
    
    index = (time_array>480) & (weight_array>0.1)

    weight_array = weight_array[index]
    fired_PMT = ChannelID[fired_PMT[index]]
    time_array = time_array[index]
    
    PMT_bins = np.hstack((-0.5,PMT_No+0.5))
    pe_array,_ = np.histogram(fired_PMT, bins=PMT_bins, weights=weight_array)
    pe_array = np.round(pe_array)
    x0_in = np.zeros(4)
    #x0_in[-1] = np.quantile(time_array,0.12)
    w_ini = np.sum(1.5*np.atleast_2d(pe_array).T*PMT_pos, axis=0)/np.sum(pe_array)/shell
    x0_in[:-1] = c2r(w_ini)
    if(x0_in[0] > 1):
        x0_in = 0.95
    result_in = minimize(Likelihood, x0_in, method='SLSQP',
                         bounds=((-1, 1), (None, None), (None, None), (None, None)), 
                         args = (PMT_pos, fired_PMT, time_array, pe_array, weight_array))
    
    in2 = r2c(result_in.x[:3])*shell
    basis = Calc_basis(result_in.x[:3], PMT_pos)
    E_in = np.sum(pe_array)/np.sum(np.exp(np.matmul(basis[:,n<=15], coef_PE)))
    print('='*80)

    print('EventID:', fin)
    '''
    print('Recon x: %+.2f m, \t Truth x: %+.2f m' %(in2[0], x[i]))
    print('Recon y: %+.2f m, \t Truth y: %+.2f m' %(in2[1], y[i]))
    print('Recon z: %+.2f m, \t Truth z: %+.2f m' %(in2[2], z[i]))
    print('Recon r: %+.2f m, \t Truth r: %+.2f m' 
          %(np.linalg.norm(in2), np.linalg.norm((x[i], y[i], z[i]))))
    print('Recon E: %+.2f, \t Truth E: %+.2f' %(E_in, E_vis[i]))
    '''
    print(in2, E_in)
    print('Likelihood: %+.2f' %(result_in.fun))
    # xyz coordinate
    recondata['x'] = in2[0]
    recondata['y'] = in2[1]
    recondata['z'] = in2[2]
    recondata['success'] = result_in.success
    recondata['Likelihood'] = result_in.fun
    recondata['t0'] = result_in.x[-1]
    recondata['E'] = E_in

    recondata['EventID'] = eval(fin)
    recondata.append()
    sys.stdout.flush()
    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.

if len(sys.argv)!=3:
    print("Wront arguments!")
    print("Usage: python Recon.py MCFileName[.root] outputFileName[.h5]")
    sys.exit(1)

# Read PMT position
PMT_pos,seq = ReadJunoPMT()

coeff_PE, coeff_time = load_coeff()
# Zernike class
cart = RZern(20)
nk = cart.nk
m = cart.mtab
n = cart.ntab

coef_time = np.zeros(nk)
coef_time[m>=0] = coeff_time

coef_PE = np.zeros(nk)
coef_PE[(m>=0) & (n<=15)] = coeff_PE
coef_PE = coef_PE[n<=15]

# Reconstruction
fin = sys.argv[1] # input file .h5
fout = sys.argv[2] # output file .h5

recon(fin, fout)
