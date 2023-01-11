from gpe.univ import my_fft
from gpe.univ.params import Params
from gpe.univ.grid import Grid
from gpe.set_device import xp


# Integral in real space
def integral(psi, grid: Grid):
    return  grid.dV * xp.sum(psi)

# Integeral in fourier space
def integral_k(psik, params):
    return params.volume * xp.sum(psik)


#Calculates the gradient of the wavefunction    
def gradient(psi, G):
    if G.params.dim == 1:
        G.U.temp[:] = my_fft.forward_transform(psi)
        G.U.Vx[:] = my_fft.inverse_transform(1j * G.grid.kx * G.U.temp) 
        
    elif G.params.dim == 2:
        G.U.temp[:, :] = my_fft.forward_transform(psi)
        G.U.Vx[:, :] = my_fft.inverse_transform(1j * G.grid.kx * G.U.temp)
        G.U.Vy[:, :] = my_fft.inverse_transform(1j * G.grid.ky * G.U.temp)
    
    elif G.params.dim == 3:
        G.U.temp[:, :, :] = my_fft.forward_transform(psi)
        G.U.Vx[:, :, :] = my_fft.inverse_transform(1j * G.grid.kx * G.U.temp)
        G.U.Vy[:, :, :] = my_fft.inverse_transform(1j * G.grid.ky * G.U.temp)
        G.U.Vz[:, :, :] = my_fft.inverse_transform(1j * G.grid.kz * G.U.temp)
    
    return