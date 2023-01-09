from gpe.univ.grid import Grid

class Params:
    """
    Class to store all necessary parameters of the simulation.
    """
    def __init__(self,
                N: list = [128, 1, 1],
                L: float = [32, 1, 1],
                g : float = 0,
                **kwargs) -> None:
             
        ## Store the parameters
        self.Nx, self.Ny, self.Nz = N
        self.Lx, self.Ly, self.Lz = L
        self.g  = g
        self.real_dtype = kwargs.get('real_dtype','np.float64')
        self.complex_dtype = kwargs.get('complex_dtype','np.complex64')
        
        ## Set the dimension
        if self.Nz == 1:
            if self.Ny == 1:
                self.dim = 1
            else:
                self.dim = 2
        else:
            self.dim = 3
            

    def __repr__(self) -> str:
        """
        returns a formated string consisting of all the paramters in the class. 
        """
        return("\n".join([
            "Parameters:",
            f"  (Nx, Ny, Nz): ({self.Nx}, {self.Ny}, {self.Nz})",
            f"  (Lx, Ly, Lz): ({self.Lx}, {self.Ly}, {self.Lz})",
            f"  g : {self.g }"])
        )




