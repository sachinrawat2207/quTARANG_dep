from quTARANG.univ.params import Params
from quTARANG.univ.grid import Grid
from quTARANG.univ import my_fft
# from gpe.set_device import xp

# Integral in real space
def integralr(params, arr, grid: Grid):
    """Integrates an array in real space (Trapezoid method).

    Parameters
    ----------
    arr : Array
    grid : Grid

    Returns
    -------
    complex
    """
    return  grid.dV * params.xp.sum(arr)

# Integeral in fourier space
def integralk(params, arrk, grid: Grid):
    """Integrates an array in real space (Trapezoid method).

    Parameters
    ----------
    arr : Array
    params : Grid

    Returns
    -------
    complex
    """
    return grid.volume * params.xp.sum(arrk)

#Calculates the gradient of the wavefunction    
def gradient(params, arr, G):
    """Calculates the gradient using spectral method along different axes depending on dimensionality.
        The result is stored in the Vx, Vy, Vz arrays inside U.
        U is an attrubite of the GPE object.

    Parameters
    ----------
    arr : Array
    G : GPE object
    
    """
    if G.params.dim == 1:
        G.U.temp[:] = my_fft.forward_transform(params, arr)
        G.U.Vx[:] = my_fft.inverse_transform(params, 1j * G.grid.kxx * G.U.temp) 
        
    elif G.params.dim == 2:
        G.U.temp[:, :] = my_fft.forward_transform(params, arr)
        G.U.Vx[:, :] = my_fft.inverse_transform(params, 1j * G.grid.kxx * G.U.temp)
        G.U.Vy[:, :] = my_fft.inverse_transform(params, 1j * G.grid.kyy * G.U.temp)
    
    elif G.params.dim == 3:
        G.U.temp[:, :, :] = my_fft.forward_transform(params, arr)
        G.U.Vx[:, :, :] = my_fft.inverse_transform(params, 1j * G.grid.kxx * G.U.temp)
        G.U.Vy[:, :, :] = my_fft.inverse_transform(params, 1j * G.grid.kyy * G.U.temp)
        G.U.Vz[:, :, :] = my_fft.inverse_transform(params, 1j * G.grid.kzz * G.U.temp)
    
    return True
