from quTARANG.univ.params import Params
# from gpe.set_device import xp

class Grid:
    def __init__(self, params: Params) -> None:
        """Stores the grid for the simulation.
        Parameters
        ----------
        params : Params object
        """
        self.setup_grid(params)
        self.setup_meshgrid(params)
        self.volume = params.volume
    
    def setup_grid(self, params: Params):
        if params.dim >= 1:
            self.setupX(params)
            self.dV = self.dx
            self.dkV = self.dkx
            
        if params.dim >= 2:
            self.setupY(params)
            self.dV *= self.dy
            self.dkV *= self.dky
            
        if params.dim == 3:
            self.setupZ(params)
            self.dV *= self.dz
            self.dkV *= self.dkz
            
    def setupX(self, params: Params):
        """Setup grid along x.
        """
        self.x = params.xp.arange(-params.Nx//2, params.Nx//2)
        self.kx = 2 * params.xp.pi * params.xp.roll(self.x, params.Nx//2)/params.Lx
        self.x = self.x * params.Lx/params.Nx
        self.dx = params.Lx/params.Nx
        self.dkx = 2 * params.xp.pi/params.Lx
    
    def setupY(self, params: Params):
        """Setup grid along y.
        """
        self.y = params.xp.arange(-params.Ny//2, params.Ny//2)
        self.ky = 2 * params.xp.pi * params.xp.roll(self.y, params.Ny//2)/params.Ly
        self.y = self.y * params.Ly/params.Ny
        self.dy = params.Ly/params.Ny
        self.dky = 2 * params.xp.pi/params.Ly

    def setupZ(self, params: Params):
        """Setup grid along x.
        """
        self.z = params.xp.arange(-params.Nz//2, params.Nz//2)
        self.kz = 2 * params.xp.pi * params.xp.roll(self.z, params.Nz//2)/params.Lz
        self.z = self.z * params.Lz/params.Nz
        self.dz  = params.Lz/params.Nz
        self.dkz = 2 * params.xp.pi/params.Lz
    
    def setup_meshgrid(self, params):
        """Generates the mesh for the simulation.
        """
        if params.dim == 1:
            self.xx = self.x
            self.kxx = self.kx
            self.ksqr = self.kxx**2 
        
        if params.dim == 2:
            self.xx, self.yy = params.xp.meshgrid(self.x, self.y, indexing = 'ij')
            self.kxx, self.kyy = params.xp.meshgrid(self.kx, self.ky, indexing = 'ij')
            self.ksqr = self.kxx**2 + self.kyy**2
        
        if params.dim == 3:
            self.xx, self.yy, self.zz = params.xp.meshgrid(self.x, self.y, self.z, indexing = 'ij')
            self.kxx, self.kyy, self.kzz = params.xp.meshgrid(self.kx, self.ky, self.kz, indexing = 'ij')
            self.ksqr = self.kxx**2 + self.kyy**2 + self.kzz**2