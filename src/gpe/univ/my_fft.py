from gpe.set_device import xp, fft

def forward_transform(arr):
    """Forward Fourier Transform
    
    Parameters
    ----------
    arr : Array
    
    Returns
    -------
    Forward Fourier transform of the input array
    """
    return fft.fftn(arr)/xp.product(arr.shape)

def inverse_transform(arr):
    """Inverse Fourier Transform
    
    Parameters
    ----------
    arr : Array
    
    Returns
    -------
    Inverse Fourier tranform of the input array
    """
    return fft.ifftn(arr) * xp.product(arr.shape)