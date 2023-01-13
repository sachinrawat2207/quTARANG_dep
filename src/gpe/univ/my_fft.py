from gpe.set_device import xp, fft

def forward_transform(arr):
    """_summary_

    Parameters
    ----------
    arr : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return fft.fftn(arr)/xp.product(arr.shape)

def inverse_transform(arr):
    """_summary_

    Parameters
    ----------
    arr : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return fft.ifftn(arr) * xp.product(arr.shape)