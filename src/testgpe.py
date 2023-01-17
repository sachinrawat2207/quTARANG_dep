from quTARANG.univ.grid import Grid
from quTARANG.univ.params import Params
from quTARANG.univ import my_fft
from quTARANG import gpe
import h5py as hp
# import numpy as xp
import cupy as xp
# from gpe.set_device import xp

N = [256, 256, 1]
L = [29.84, 29.84, 1]

par = Params(N, L, g = 2, dt=0.0001, tmax = 0.0002, gpu=True)
G = gpe.GPE(par)

## define the functional form of the initial wavefunction and potential
def wfc(x, y=0, z=0):
    eps1 = 0.25
    const = 8**0.25/(xp.pi * eps1)**(3/4)
    return const * xp.exp(-(x**2 + 2*y**2 + 4*z**2)/(2*eps1))
    
def pot(x, y=0, z=0):
    return 0.5*(x**2 + 4*y**2 + 16*z**2)

print(par)
G.set_init(wfc, pot)
print(type(G.wfc))

# G.evolve()
# xp.random.seed(0)

# for i in range(1):
    
#     comp, incomp = G.KE_decomp()
#     print(G.compute_energy())
#     print(G.compute_kinetic_energy(), G.compute_quantum_energy(), G.compute_internal_energy())
#     print(comp, incomp)    
#     print('\n\n\n')
#     G.evolve_ms()
# print(type(G.wfc), G.wfc.shape)

# f = hp.f = hp.File(par.path/'rms.hdf5', 'r')
# xrms = xp.array(f['xrms'])
# t = xp.array(f['t'])
# f.close()