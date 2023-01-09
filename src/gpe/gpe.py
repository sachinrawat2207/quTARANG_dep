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
    
    def compute_chempot(self):   #C
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
    
'''
    def comp_energy(self, U = Vector_field()): #C
        U.temp[:] = my_fft.forward_transform(self.wfc) #for sstep_strang
        deriv = self.params.volume * xp.sum(my_fft.ksqr * xp.abs(U.temp)**2)  
        return fns.integral(((self.pot + 0.5 * self.params.g * xp.abs(self.wfc)**2) * xp.abs(self.wfc)**2)) + deriv/2
    
    def comp_quantum_energy(self, U = Vector_field()): #C
        fns.gradient(xp.abs(self.wfc), U)
        if self.params.dim == 2:
            U.temp[:] = 0.5 * (U.Vx**2 + U.Vy**2) 

        elif self.params.dim == 3:
            U.temp[:] = 0.5 * (U.Vx**2 + U.Vy**2 + U.Vz**2)
        return fns.integral(U.temp.real)


    def comp_internal_energy(self, U = Vector_field()): #C
        return 0.5 * para.g * fns.integral(xp.abs(self.wfc)**4)
        

    def comp_potential_energy(self, U = Vector_field()):   #C
        return fns.integral(self.pot * xp.abs(self.wfc)**2) 
        
    def comp_velocity(self, U = Vector_field()): #C
        fns.gradient(self.wfc.conj(), U)
        if para.dim == 2:
            U.Vx[:] = -(self.wfc * U.Vx).imag/xp.abs(self.wfc)**2 
            U.Vy[:] = -(self.wfc * U.Vy).imag/xp.abs(self.wfc)**2 
    
        elif para.dim == 3:
            U.Vx[:] = -(self.wfc * U.Vx).imag/xp.abs(self.wfc)**2 
            U.Vy[:] = -(self.wfc * U.Vy).imag/xp.abs(self.wfc)**2 
            U.Vz[:] = -(self.wfc * U.Vz).imag/xp.abs(self.wfc)**2 
        return 
    

    def comp_kinetic_energy(self, U = Vector_field): #C doubt while integrating in fourier space
        self.comp_velocity(U)
        U.temp[:] = 0.5 * xp.abs(self.wfc)**2 * (U.Vx**2 + U.Vy**2 + U.Vz**2)
        return fns.integral(U.temp.real)

     
    def omegak(self, U = Vector_field()):  #C
        self.comp_velocity(U) 
        U.temp[:] = xp.abs(self.wfc)
        U.omegai_kx[:] = my_fft.forward_transform(U.temp * U.Vx)
        U.omegai_ky[:] = my_fft.forward_transform(U.temp * U.Vy)
        U.omegai_kz[:] = my_fft.forward_transform(U.temp * U.Vz)
        if para.dim == 2:
            my_fft.ksqr[0, 0] = 1
            U.temp[:] = (my_fft.kx * U.omegai_kx + my_fft.ky * U.omegai_ky)/my_fft.ksqr
            # Compressible part calculation
            U.Vx[:] = my_fft.kx * U.temp
            U.Vy[:] = my_fft.ky * U.temp       
            my_fft.ksqr[0, 0] = 0   
            U.omegai_kx[:] = U.omegai_kx - U.Vx
            U.omegai_ky[:] = U.omegai_ky - U.Vy
        else:
            my_fft.ksqr[0, 0, 0] = 1 
            U.temp[:] = (my_fft.kx * U.omegai_kx + my_fft.ky * U.omegai_ky + my_fft.kz * U.omegai_kz)/my_fft.ksqr
            # Compressible part calculation
            U.Vx[:] = my_fft.kx * U.temp
            U.Vy[:] = my_fft.ky * U.temp
            U.Vz[:] = my_fft.kz * U.temp
            my_fft.ksqr[0, 0, 0] = 0 
            U.omegai_kx[:] = U.omegai_kx - U.Vx
            U.omegai_ky[:] = U.omegai_ky - U.Vy
            U.omegai_kz[:] = U.omegai_kz - U.Vz
        return 


    def KE_decomp(self, U = Vector_field()): #C
        self.omegak(U)
        if para.dim == 2:
            KE_comp = 0.5 * fns.integral_k(xp.abs(U.Vx)**2 + xp.abs(U.Vy)**2)
            KE_incomp = 0.5 * fns.integral_k(xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2)
        else:
            KE_comp = 0.5 * fns.integral_k(xp.abs(U.Vx)**2 + xp.abs(U.Vy)**2 + xp.abs(U.Vz)**2)
            KE_incomp = 0.5 * fns.integral_k(xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2 + xp.abs(U.omegai_kz)**2)
        return KE_comp, KE_incomp 
        
    
        # For calculation of particle number flux
    def comp_tk_particle_no(self, U = Vector_field()):    
        U.temp[:] = my_fft.forward_transform(self.wfc)
        U.temp1[:] = my_fft.forward_transform(para.g * self.wfc * xp.abs(self.wfc)**2 + self.wfc * self.pot)
        temp = (U.temp1[:] * xp.conjugate(U.temp)).imag
        return self.binning(temp)
    
    def comp_par_no_spectrum(self, U = Vector_field()):
        U.temp[:] = xp.abs(my_fft.forward_transform(self.wfc))**2
        return self.bining(U.temp)
        
    def comp_KEcomp_spectrum(self, U = Vector_field()):
        self.omegak(U)
        if para.dim == 2:
            KE_incompk = 0.5 * (xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2) 
            KE_compk = 0.5 * (xp.abs(U.Vx)**2 + xp.abs(U.Vy)**2)
        elif para.dim == 3:
            KE_incompk = 0.5 * (xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2 + xp.abs(U.omegai_kz)**2) 
            KE_compk = 0.5 * (xp.abs(U.Vx)**2 + xp.abs(U.Vy)**2 + xp.abs(U.Vz)**2)
        KEcomp_spectrum = self.binning(KE_compk)
        KEincomp_spectrum = self.binning(KE_incompk)
        return KEcomp_spectrum, KEincomp_spectrum
    
    
    def comp_QE_spectrum(self, U = Vector_field()):
        U.temp[:] = fns.gradient(xp.abs(self.wfc), U)
        U.temp[:] = my_fft.forward_transform(U.temp, axes = U.temp.shape[1:])
        U.temp[:] = xp.sum(xp.abs(U.temp)**2, axis = 0)
        QE_spectrum = self.binning(U.temp)
        return QE_spectrum
    
    
    def comp_IE_spectrum(self, U = Vector_field()):
        U.temp[:] = my_fft.forward_transform(xp.abs(self.wfc)**2)
        U.temp[:] = 0.5 * para.g * xp.abs(U.temp)**2
        IE_spectrum = self.binning(U.temp)
        return IE_spectrum   
'''