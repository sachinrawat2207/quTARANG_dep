from gpe.univ import my_fft
from gpe.univ.params import Params
from gpe.univ.grid import Grid
from gpe.set_device import xp

# Integral in real space
def integralr(arr, grid: Grid):
    """Integrates an array in real space (Trapezoid method).

    Parameters
    ----------
    arr : Array
    grid : Grid

    Returns
    -------
    complex
    """
    return  grid.dV * xp.sum(arr)

# Integeral in fourier space
def integralk(arrk, grid: Grid):
    """Integrates an array in real space (Trapezoid method).

    Parameters
    ----------
    arr : Array
    params : Grid

    Returns
    -------
    complex
    """
    return grid.volume * xp.sum(arrk)

#Calculates the gradient of the wavefunction    
def gradient(arr, G):
    """Calculates the gradient using spectral method along different axes depending on dimensionality.
        The result is stored in the Vx, Vy, Vz arrays inside U.
        U is an attrubite of the GPE object.

    Parameters
    ----------
    arr : Array
    G : GPE object
    
    """
    if G.params.dim == 1:
        G.U.temp[:] = my_fft.forward_transform(arr)
        G.U.Vx[:] = my_fft.inverse_transform(1j * G.grid.kx * G.U.temp) 
        
    elif G.params.dim == 2:
        G.U.temp[:, :] = my_fft.forward_transform(arr)
        G.U.Vx[:, :] = my_fft.inverse_transform(1j * G.grid.kx * G.U.temp)
        G.U.Vy[:, :] = my_fft.inverse_transform(1j * G.grid.ky * G.U.temp)
    
    elif G.params.dim == 3:
        G.U.temp[:, :, :] = my_fft.forward_transform(arr)
        G.U.Vx[:, :, :] = my_fft.inverse_transform(1j * G.grid.kx * G.U.temp)
        G.U.Vy[:, :, :] = my_fft.inverse_transform(1j * G.grid.ky * G.U.temp)
        G.U.Vz[:, :, :] = my_fft.inverse_transform(1j * G.grid.kz * G.U.temp)
    
    return True
