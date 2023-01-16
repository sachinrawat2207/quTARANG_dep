from gpe.univ.params import Params
# from gpe.set_device import xp, fft

def forward_transform(params, arr):
    """Forward Fourier Transform
    
    Parameters
    ----------
    arr : Array
    
    Returns
    -------
    Forward Fourier transform of the input array
    """
    return params.xp.fft.fftn(arr)/params.xp.product(arr.shape)

def inverse_transform(params, arr):
    """Inverse Fourier Transform
    
    Parameters
    ----------
    arr : Array
    
    Returns
    -------
    Inverse Fourier tranform of the input array
    """
    return params.xp.fft.ifftn(arr) * params.xp.product(arr.shape)