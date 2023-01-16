# from gpe.set_device import set_gpu
from pathlib import Path

class Params:
    """ Class to store all necessary parameters of the simulation.
    """
    def __init__(self,
                N: list = [128, 1, 1],
                L: float = [32, 1, 1],
                g: float = 0,
                dt: float = 0.001,
                tmax: float = 1,
                scheme: str = 'TSSP',
                imgtime: bool = False,
                rms: list = [False, 0, 100],
                ektk: list = [False, 0, 100],
                energy: list = [False, 0, 100],
                wfc: list = [False, 0, 100],
                gpu: bool = False,
                gpu_rank: int = 0,
                path: str = str(Path.cwd()/'output'),
                **kwargs) -> None:
        """
        Parameters
        ----------
        N : list, optional
            Grid points along x, y, z, by default [128, 1, 1]
        L : float, optional
            Length along x, y, z, by default [32, 1, 1]
        g : float, optional
            Nonlinearity constant, by default 0
        dt : float, optional
            time step, by default 0.001
        tmax : float, optional
            Maximum time , by default 1
        scheme : str, optional
            scheme, by default 'TSSP'
        imgtime : bool, optional
            set true for imaginary time, by default False

        """
        ## Store the parameters
        self.Nx, self.Ny, self.Nz = N
        self.Lx, self.Ly, self.Lz = L
        self.g  = g
        self.real_dtype = kwargs.get('real_dtype','float64')
        self.complex_dtype = kwargs.get('complex_dtype','complex128')
        self.dt = dt
        self.tmax = tmax
        self.volume = self.Lx * self.Ly * self.Lz
        self.scheme = scheme
        self.imgtime = imgtime
        self.gpu = gpu
        self.gpu_rank = gpu_rank
        
        self.path = Path(path)
        self.save_energy, self.save_en_start_step, self.save_en_iter_step = energy
        self.save_rms, self.save_rms_start_step, self.save_rms_iter_step = rms
        self.save_ektk, self.save_ektk_start_step, self.save_ektk_iter_step  =  ektk
        self.save_wfc, self.save_wfc_start_step, self.save_wfc_iter_step = wfc
        
        if self.gpu == True:
            import cupy as cp
            import cupy.fft as fft
            dev = cp.cuda.Device(self.gpu_rank)
            dev.use()
            self.xp = cp
            self.fft = fft
        else:
            import numpy as np
            from pyfftw.interfaces import numpy_fft
            self.xp = np
            self.fft = numpy_fft
        
        ## Set the dimension
        if self.Nz == 1:
            if self.Ny == 1:
                self.dim = 1
            else:
                self.dim = 2
        else:
            self.dim = 3
        
        self.nstep = int(tmax/dt)    


    def __repr__(self) -> str:
        """
        returns a formated string consisting of all the paramters in the class. 
        """
        return("\n".join([
            "Parameters:",
            f"  dim : {self.dim}",
            f"  (Nx, Ny, Nz): ({self.Nx}, {self.Ny}, {self.Nz})",
            f"  (Lx, Ly, Lz): ({self.Lx}, {self.Ly}, {self.Lz})",
            f"  Volume : {self.volume}",
            f"  g : {self.g}",
            f"  dt: {self.dt}",
            f"  tmax: {self.tmax}",
            f"  Numerical Scheme: {self.scheme}",
            f"  Imaginary Time: {self.imgtime}"
            ])
        )

