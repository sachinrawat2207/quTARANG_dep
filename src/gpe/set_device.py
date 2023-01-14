import numpy as xp
import pyfftw.interfaces.numpy_fft as fft

# from cupyx.fallback_mode import numpy as xp
# fft = xp.fft
# from cupyx.fallback_mode.numpy import fft

def set_gpu(rank=0):
    """Initial setup for GPU run. Sets xp to cupy, fft to cupy.fft and fixes which GPU to use when multiple GPUs are available.

    Parameters
    ----------
    rank : int, optional
        Rank of the gpu device to be used when multiple GPUs are available, by default 0
    """
    import cupy as xp
    import cupy.fft as fft
    dev = xp.cuda.Device(rank)
    dev.use()
    # import cupy.core._accelerator as _acc
    # _acc.set_routine_accelerators(['cub'])
    # _acc.set_reduction_accelerators(['cub'])
