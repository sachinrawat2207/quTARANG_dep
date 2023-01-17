from quTARANG.univ.grid import Grid
from quTARANG.univ.params import Params
# from gpe.set_device import xp
from quTARANG.univ import my_fft
from quTARANG import gpe
import h5py as hp
import numpy as np
import matplotlib.pyplot as plt

## define the functional form of the initial wavefunction and potential
def wfc(x, y, z):
    eps1 = 0.25
    const = 8**0.25/(np.pi * eps1)**(3/4)
    return const * np.exp(-(x**2 + 2*y**2 + 4*z**2)/(2*eps1))
    
def pot(x, y=0, z=0):
    return 0.5*(x**2 + 4*y**2 + 16*z**2)

## Add simulation parameters to an instance of the Params class
N = [64, 64, 64]
L = [16, 16, 16]
g = 0.1
par = Params(N, L, g, dt=0.001, tmax = 2, rms = [True, 0, 100])

## Initiate the GPE class and run the simulation
G = gpe.GPE(par)
G.set_init(wfc, pot)

G.evolve()
path1 = str(par.path/'output/rms.hdf5')

## Load the generated and reference results
f = hp.File(path1, 'r') ## fix path
t = np.array(f['t_rms'])
xrms = np.array(f['xrms'])
yrms = np.array(f['yrms'])
zrms = np.array(f['zrms'])
f.close()

path='/home/sachinr/Desktop/test/bao_data/'
t1, xrms_ref = np.loadtxt(path + '3D_I_a.txt', delimiter = ',', unpack=True)
t2, yrms_ref = np.loadtxt(path + '3D_I_b.txt', delimiter = ',', unpack=True)
t3, zrms_ref = np.loadtxt(path + '3D_I_c.txt', delimiter = ',', unpack=True)

## Plotting
plt.rcParams.update({'font.size':'20'})
plt.figure(figsize=(7, 7))
plt.title("3D case I")
plt.xlim(t[0], t[-1])
plt.ylim()
plt.ylabel("Condensate width ($\sigma$)")
plt.xlabel("t")
plt.plot(t, xrms, label='$\sigma_x$')
plt.plot(t, yrms, label='$\sigma_y$')
plt.plot(t, zrms, label='$\sigma_z$')
plt.scatter(t1, xrms_ref, s = 20, label = "$\sigma_x$(ref)")
plt.scatter(t2, yrms_ref, s = 20, label = "$\sigma_y$(ref)")
plt.scatter(t3, zrms_ref, s = 20, label = "$\sigma_z$(ref)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

