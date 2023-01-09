from gpe.set_device import xp
from gpe.univ.my_fft import *
from gpe.univ.grid import Grid
from gpe.univ.params import Params
from gpe.univ import my_fft



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
        if params.dimension == 2:
            self.Vx = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.Vy = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.Vz = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_kx = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_ky = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_kz = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.temp = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.temp1 = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
 
        if params.dimension == 3:
            self.Vx = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.Vy = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.Vz = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_kx = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_ky = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_kz = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.temp = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.temp1 = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
    


class GPE():
    def __init__(self, params: Params, grid: Grid):
        self.grid = Grid(params)
        self.wfc = []
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
        self.set_arrays()
        
    def set_arrays(self, params: Params):   
        if params.dimension == 2:
            self.wfc = xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.V = xp.zeros((params.Nx, params.Ny), dtype = params.real_dtype)
            
        elif params.dimension == 3:
            self.wfc = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.V = xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.real_dtype)
    
    
    
    def comp_energy(self, U = Vector_field()): #C
        U.temp[:] = my_fft.forward_transform(self.wfc) #for sstep_strang
        deriv = para.volume * xp.sum(my_fft.ksqr * xp.abs(U.temp)**2)  
        return fns.integral(((self.V + 0.5 * para.g * xp.abs(self.wfc)**2) * xp.abs(self.wfc)**2)) + deriv/2
    
    def comp_quantum_energy(self, U = Vector_field()): #C
        fns.gradient(xp.abs(self.wfc), U)
        if para.dimension == 2:
            U.temp[:] = 0.5 * (U.Vx**2 + U.Vy**2) 

        elif para.dimension == 3:
            U.temp[:] = 0.5 * (U.Vx**2 + U.Vy**2 + U.Vz**2)
        return fns.integral(U.temp.real)


    def comp_internal_energy(self, U = Vector_field()): #C
        return 0.5 * para.g * fns.integral(xp.abs(self.wfc)**4)
        

    def comp_potential_energy(self, U = Vector_field()):   #C
        return fns.integral(self.V * xp.abs(self.wfc)**2) 
        
    def comp_velocity(self, U = Vector_field()): #C
        fns.gradient(self.wfc.conj(), U)
        if para.dimension == 2:
            U.Vx[:] = -(self.wfc * U.Vx).imag/xp.abs(self.wfc)**2 
            U.Vy[:] = -(self.wfc * U.Vy).imag/xp.abs(self.wfc)**2 
    
        elif para.dimension == 3:
            U.Vx[:] = -(self.wfc * U.Vx).imag/xp.abs(self.wfc)**2 
            U.Vy[:] = -(self.wfc * U.Vy).imag/xp.abs(self.wfc)**2 
            U.Vz[:] = -(self.wfc * U.Vz).imag/xp.abs(self.wfc)**2 
        return 
    
    def norm(self): #C
        return fns.integral(xp.abs(self.wfc)**2)

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
        if para.dimension == 2:
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
        if para.dimension == 2:
            KE_comp = 0.5 * fns.integral_k(xp.abs(U.Vx)**2 + xp.abs(U.Vy)**2)
            KE_incomp = 0.5 * fns.integral_k(xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2)
        else:
            KE_comp = 0.5 * fns.integral_k(xp.abs(U.Vx)**2 + xp.abs(U.Vy)**2 + xp.abs(U.Vz)**2)
            KE_incomp = 0.5 * fns.integral_k(xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2 + xp.abs(U.omegai_kz)**2)
        return KE_comp, KE_incomp


    # def rms(self, quantity, U = Vector_field()):  #C
    #     U.temp[:] = xp.abs(self.wfc)**2 * quantity**2
    #     return (fns.integral(U.temp.real))**0.5

    def comp_zrms(self):   
        return (fns.integral(xp.abs(self.wfc) ** 2 * my_fft.z ** 2) - (fns.integral(xp.abs(self.wfc) ** 2 * my_fft.z))**2)**.5     
       
    def comp_xrms(self):
        return (fns.integral(xp.abs(self.wfc)**2 * my_fft.x**2) - (fns.integral(xp.abs(self.wfc)**2 * my_fft.x))**2)**.5
        
    def comp_yrms(self):
        return (fns.integral(xp.abs(self.wfc)**2 * my_fft.y**2) - (fns.integral(xp.abs(self.wfc)**2 * my_fft.y))**2)**.5    
          
    def chempot(self, U = Vector_field()):   #C
        U.temp[:] = my_fft.forward_transform(self.wfc) #for sstep_strang
        deriv = para.volume * xp.sum(my_fft.ksqr * xp.abs(U.temp)**2)  
        return fns.integral(((self.V + para.g * xp.abs(self.wfc)**2) * xp.abs(self.wfc)**2)) + deriv/2
    
        # For calculation of particle number flux
    def comp_tk_particle_no(self, U = Vector_field()):    
        U.temp[:] = my_fft.forward_transform(self.wfc)
        U.temp1[:] = my_fft.forward_transform(para.g * self.wfc * xp.abs(self.wfc)**2 + self.wfc * self.V)
        temp = (U.temp1[:] * xp.conjugate(U.temp)).imag
        return self.binning(temp)
    
    def comp_par_no_spectrum(self, U = Vector_field()):
        U.temp[:] = xp.abs(my_fft.forward_transform(self.wfc))**2
        return self.bining(U.temp)
        
    def comp_KEcomp_spectrum(self, U = Vector_field()):
        self.omegak(U)
        if para.dimension == 2:
            KE_incompk = 0.5 * (xp.abs(U.omegai_kx)**2 + xp.abs(U.omegai_ky)**2) 
            KE_compk = 0.5 * (xp.abs(U.Vx)**2 + xp.abs(U.Vy)**2)
        elif para.dimension == 3:
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
    
    
