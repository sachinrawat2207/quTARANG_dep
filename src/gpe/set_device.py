import numpy as xp
import pyfftw.interfaces.numpy_fft as fft

def set_gpu(rank = 0):
    import cupy as xp
    import cupy.fft as fft
    dev = xp.cuda.Device(rank)
    dev.use()
    # import cupy.core._accelerator as _acc
    # _acc.set_routine_accelerators(['cub'])
    # _acc.set_reduction_accelerators(['cub'])