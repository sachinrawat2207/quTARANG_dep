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


# Calculates the gradient of the wavefunction
# def gradient(psi, params: Params, grid: Grid, U = Vector_field()):
#     if params.dim == 1:
#         U.temp[:] = my_fft.forward_transform(psi)
#         U.Vx = my_fft.inverse_transform(1j * grid.kx * U.temp) 
        
#     elif params.dim == 2:
#         U.temp[:, :] = my_fft.forward_transform(psi)
#         U.Vx[:, :] = my_fft.inverse_transform(1j * grid.kx * U.temp)
#         U.Vy[:, :] = my_fft.inverse_transform(1j * grid.ky * U.temp)
    
#     elif params.dim == 3:
#         U.temp[:, :, :] = my_fft.forward_transform(psi)
#         U.Vx[:, :, :] = my_fft.inverse_transform(1j * grid.kx * U.temp)
#         U.Vy[:, :, :] = my_fft.inverse_transform(1j * grid.ky * U.temp)
#         U.Vz[:, :, :] = my_fft.inverse_transform(1j * grid.kz * U.temp)
    
#     return