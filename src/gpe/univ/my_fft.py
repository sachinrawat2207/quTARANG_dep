from gpe.set_device import xp, fft

def forward_transform(arr):
    return fft.fftn(arr)/xp.product(arr.shape)

def inverse_transform(arr):
    return fft.ifftn(arr) * xp.product(arr.shape)