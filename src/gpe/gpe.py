from gpe.set_device import xp
from gpe.univ.my_fft import *
from gpe.univ.grid import Grid
from gpe.univ.params import Params
from gpe.univ import my_fft, fns
from gpe import evolution

class Vector_field():
    Vx = []
    Vx = []  
    Vy = []
    Vz = []
    # Used to take omegai
    omegai_kx = [] 
    omegai_ky = []
    omegai_kz = []
    # Temporary variable
    temp = []
    
    def __init__(self, params: Params):
        self.set_arrays(params)
        
    def set_arrays(self, params: Params):
        if params.dim == 2:
            self.Vx = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.Vy = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.Vz = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_kx = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_ky = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_kz = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.temp = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.temp1 = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
        
        if params.dim == 3:
            self.Vx = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.Vy = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.Vz = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_kx = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_ky = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_kz = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.temp = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.temp1 = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)

class GPE():
    def __init__(self, params: Params, set_init):
        self.params = params
        self.grid = Grid(params)
        self.wfc = []
        self.wfck = []
        self.Npar = 0
        self.pot = []
        self.comp_KE = []
        self.incomp_KE = []
        self.quantum_energy = []
        self.internal_energy = []
        self.total_energy = []
        self.total_KE = []
        self.potential_energy = []
        self.t_energy = []
        
        self.KEcomp_spec = []
        self.KEincomp_spec = []
        self.tk_par_no = []
        self.t_ektk = []
        
        self.xrms = []
        self.yrms = []
        self.zrms = []
        self.t_rms = []
        
        self.U = Vector_field(params)
        self.set_arrays()
        self.set_init(set_init)
        evolution.set_scheme(self)
        
    def set_arrays(self):
        if self.params.dim == 1:
            self.wfc = xp.zeros((self.params.Nx), dtype = self.params.complex_dtype)
            self.wfck = xp.zeros((self.params.Nx), dtype = self.params.complex_dtype)
            self.pot = xp.zeros((self.params.Nx), dtype = self.params.real_dtype)
            
        elif self.params.dim == 2:
            self.wfc = xp.zeros((self.params.Nx, self.params.Ny), dtype = self.params.complex_dtype)
            self.wfck = xp.zeros((self.params.Nx, self.params.Ny), dtype = self.params.complex_dtype)
            self.pot = xp.zeros((self.params.Nx, self.params.Ny), dtype = self.params.real_dtype)
        
        else:
            self.wfc = xp.zeros((self.params.Nx, self.params.Ny, self.params.Nz), dtype = self.params.complex_dtype)
            self.pot = xp.zeros((self.params.Nx, self.params.Ny, self.params.Nz), dtype = self.params.real_dtype)
    
    def set_init(self, fun):
        if self.params.dim == 1:
            self.wfc[:], self.pot[:] = fun(self.grid.xx)
        
        if self.params.dim == 2:
            self.wfc[:], self.pot[:] = fun(self.grid.xx, self.grid.yy)
        
        if self.params.dim == 3:
            self.wfc[:], self.pot[:] = fun(self.grid.xx, self.grid.yy, self.grid.zz)
        self.Npar = self.compute_norm()
    
    def compute_norm(self):
        return fns.integral(xp.abs(self.wfc)**2, self.grid)**0.5
    
    def renormalize(self, normFact: float = 1.0):
        self.wfc = normFact * self.wfc/self.compute_norm()
    
    def evolve(self):
        evolution.time_advance(self)
    
    def evolve_ms(self):
        evolution.time_advance_ms(self)    
    
    def compute_chempot(self):   
        self.U.temp[:] = my_fft.forward_transform(self.wfc) #for sstep_strang
        deriv = self.params.volume * xp.sum(self.grid.ksqr * xp.abs(self.U.temp)**2)  
        return fns.integral(((self.pot + self.params.g * xp.abs(self.wfc)**2) * xp.abs(self.wfc)**2), self.grid) + deriv/2
    
    def compute_xrms(self):
        return (fns.integral(xp.abs(self.wfc)**2 * self.grid.xx**2, self.grid) - (fns.integral(xp.abs(self.wfc)**2 * self.grid.xx, self.grid))**2)**.5
        
    def compute_yrms(self):
        return (fns.integral(xp.abs(self.wfc)**2 * self.grid.yy**2, self.grid) - (fns.integral(xp.abs(self.wfc)**2 * self.grid.yy, self.grid))**2)**.5   

    def compute_zrms(self):   
        return (fns.integral(xp.abs(self.wfc)**2 * self.grid.zz**2, self.grid) - (fns.integral(xp.abs(self.wfc)**2 * self.grid.zz, self.grid))**2)**.5
    
    def compute_rrms(self):
        if self.params.dim == 2:
            return (fns.integral(xp.abs(self.wfc) ** 2 * (self.grid.xx**2 + self.grid.yy**2), self.grid))**.5
        return (fns.integral(xp.abs(self.wfc) ** 2 * (self.grid.xx**2 + self.grid.yy**2 + self.grid.zz**2), self.grid))**.5
    
    # def compute_rrms(self):   
    #     return (fns.integral(xp.abs(self.wfc)**2 * (self.grid.xx**2 + self.grid.yy**2 + self.grid.zz**2), self.grid) - (fns.integral(xp.abs(self.wfc)**2 * self.grid.zz, self.grid))**2)**.5

    def compute_energy(self):     #C
        self.self.U.temp[:] = my_fft.forward_transform(self.wfc) #for sstep_strang
        deriv = self.params.volume * xp.sum(self.grid.ksqr * xp.abs(self.U.temp)**2)  
        return fns.integral(((self.pot + 0.5 * self.params.g * xp.abs(self.wfc)**2) * xp.abs(self.wfc)**2), self.grid) + deriv/2
    
    def compute_quantum_energy(self):  #C 
        fns.gradient(xp.abs(self.wfc), self)
        if self.params.dim == 2:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2) 

        elif self.params.dim == 3:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        return fns.integral(self.U.temp.real, self.grid)


    def comp_internal_energy(self):  #C 
        return 0.5 * para.g * fns.integral(xp.abs(self.wfc)**4, self.grid)
        

    def comp_potential_energy(self): #C   
        return fns.integral(self.pot * xp.abs(self.wfc)**2, self.grid) 
        
    def comp_velocity(self):   #C
        fns.gradient(self.wfc.conj(), self)
        if self.params.dim == 2:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/xp.abs(self.wfc)**2 
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/xp.abs(self.wfc)**2 
    
        elif self.params.dim == 3:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/xp.abs(self.wfc)**2 
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/xp.abs(self.wfc)**2 
            self.U.Vz[:] = -(self.wfc * self.U.Vz).imag/xp.abs(self.wfc)**2 
        return 
    

    def comp_kinetic_energy(self):  # doubt while integrating in fourier space
        self.comp_velocity()
        self.U.temp[:] = 0.5 * xp.abs(self.wfc)**2 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        return fns.integral(self.U.temp.real, self.grid)

     
    def omegak(self):   #C
        self.comp_velocity() 
        self.U.temp[:] = xp.abs(self.wfc)
        U.omegai_kx[:] = my_fft.forward_transform(self.U.temp * self.U.Vx)
        U.omegai_ky[:] = my_fft.forward_transform(self.U.temp * self.U.Vy)
        U.omegai_kz[:] = my_fft.forward_transform(self.U.temp * self.U.Vz)
        if self.params.dim == 2:
            self.grid.ksqr[0, 0] = 1
            self.U.temp[:] = (self.grid.kxx * U.omegai_kx + self.grid.kyy * U.omegai_ky)/self.grid.ksqr
            # Compressible part calculation
            self.U.Vx[:] = self.grid.kxx * self.U.temp
            self.U.Vy[:] = self.grid.kyy * self.U.temp       
            self.grid.ksqr[0, 0] = 0   
            U.omegai_kx[:] = U.omegai_kx - self.U.Vx
            U.omegai_ky[:] = U.omegai_ky - self.U.Vy
        else:
            self.grid.ksqr[0, 0, 0] = 1 
            self.U.temp[:] = (self.grid.kxx * U.omegai_kx + self.grid.kyy * U.omegai_ky + self.grid.kzz * U.omegai_kz)/self.grid.ksqr
            # Compressible part calculation
            self.U.Vx[:] = self.grid.kxx * self.U.temp
            self.U.Vy[:] = self.grid.kyy * self.U.temp
            self.U.Vz[:] = self.grid.kzz * self.U.temp
            self.grid.ksqr[0, 0, 0] = 0 
            U.omegai_kx[:] = U.omegai_kx - self.U.Vx
            U.omegai_ky[:] = U.omegai_ky - self.U.Vy
            U.omegai_kz[:] = U.omegai_kz - self.U.Vz
        return 


    def KE_decomp(self):   #C
        self.omegak()
        if self.params.dim == 2:
            KE_comp = 0.5 * fns.integral_k(xp.abs(self.U.Vx)**2 + xp.abs(self.U.Vy)**2, self.params)
            KE_incomp = 0.5 * fns.integral_k(xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2, self.params)
        else:
            KE_comp = 0.5 * fns.integral_k(xp.abs(self.U.Vx)**2 + xp.abs(self.U.Vy)**2 + xp.abs(self.U.Vz)**2, self.params)
            KE_incomp = 0.5 * fns.integral_k(xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2 + xp.abs(U.omegai_kz)**2, self.params)
        return KE_comp, KE_incomp 
        
    
    # For calculation of particle number flux
    def comp_tk_particle_no(self):    
        self.U.temp[:] = my_fft.forward_transform(self.wfc)
        self.U.temp1[:] = my_fft.forward_transform(para.g * self.wfc * xp.abs(self.wfc)**2 + self.wfc * self.pot)
        temp = (self.U.temp1[:] * xp.conjugate(self.U.temp)).imag
        return self.binning(temp)
    
    def comp_par_no_spectrum(self):
        self.U.temp[:] = xp.abs(my_fft.forward_transform(self.wfc))**2
        return self.bining(self.U.temp)
        
    def comp_KEcomp_spectrum(self):
        self.omegak()
        if self.params.dim == 2:
            KE_incompk = 0.5 * (xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2) 
            KE_compk = 0.5 * (xp.abs(self.U.Vx)**2 + xp.abs(self.U.Vy)**2)
        elif self.params.dim == 3:
            KE_incompk = 0.5 * (xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2 + xp.abs(U.omegai_kz)**2) 
            KE_compk = 0.5 * (xp.abs(self.U.Vx)**2 + xp.abs(self.U.Vy)**2 + xp.abs(self.U.Vz)**2)
        KEcomp_spectrum = self.binning(KE_compk)
        KEincomp_spectrum = self.binning(KE_incompk)
        return KEcomp_spectrum, KEincomp_spectrum
    
    def comp_IE_spectrum(self):
        self.U.temp[:] = my_fft.forward_transform(xp.abs(self.wfc)**2)
        self.U.temp[:] = 0.5 * para.g * xp.abs(self.U.temp)**2
        IE_spectrum = self.binning(self.U.temp)
        return IE_spectrum   
