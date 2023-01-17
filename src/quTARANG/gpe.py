from quTARANG.univ.grid import Grid
from quTARANG.univ.params import Params
from quTARANG.univ import my_fft, fns
from quTARANG import evolution
# from gpe.set_device import xp
# from gpe.univ.my_fft import *

class Vector_field():
    """ It contains the variables which helps to calculate the different physical quantities of GPE class 
    """
    Vx = []
    Vx = []  
    Vy = []
    Vz = []
    
    # Used to calculate omegai
    omegai_kx = []
    omegai_ky = []
    omegai_kz = []
    
    # Temporary variable
    temp = []
    
    def __init__(self, params: Params):
        self.set_arrays(params)
        
    def set_arrays(self, params: Params):
        if params.dim == 2:
            self.Vx = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.Vy = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.Vz = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_kx = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_ky = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.omegai_kz = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.temp = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
            self.temp1 = params.xp.zeros((params.Nx, params.Ny), dtype = params.complex_dtype)
        
        if params.dim == 3:
            self.Vx = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.Vy = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.Vz = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_kx = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_ky = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.omegai_kz = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.temp = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)
            self.temp1 = params.xp.zeros((params.Nx, params.Ny, params.Nz), dtype = params.complex_dtype)

class GPE():
    """Main class for the simulation. 
    """
    def __init__(self, params: Params):
        """Intialisation of the GPE class.

        Parameters
        ----------
        params : Params
        
        Attributes
        ----------
        params : an instance of the Params class for simulation
        grid : Sets up the grid using Grid class
        wfc : Wavefunction
        wfck : Fourier tranform of the wavefunction
        pot : Potential of the system
        Npar : Number of particles
        total_energy : Total energy of the system
        total_KE : Total kinetic energy of the system
        quantum_energy : Quantum pressure energy
        potential_energy : Potential energy
        internal_energy : Internal/Interaction energy
        comp_KE : compressible kinetic energy
        incomp_KE : incompressible kinetic energy
        t_energy : Time at which energy is calculated
        KEcomp_spec : Spectrum of compressible kinetic energy
        KEincomp_spec : Spectrum of incompressible kinetic energy
        t_ektk : time to save ektk(energy spectr)
        xrms : root mean square size of the condensate along the x direction
        yrms : root mean square size of the condensate along the y direction
        zrms : root mean square size of the condensate along the z direction
        t_rms : Time at which rms values are calculating
        """

        self.params = params
        self.grid = Grid(params)
        self.wfc = []
        self.wfck = []
        self.pot = []
        self.Npar = 0
        self.total_energy = []
        self.total_KE = []
        self.quantum_energy = []
        self.potential_energy = []
        self.internal_energy = []
        self.comp_KE = []
        self.incomp_KE = []
        self.t_energy = []
        
        self.KEcomp_spec = []
        self.KEincomp_spec = []
        self.t_ektk = []
        
        self.xrms = []
        self.yrms = []
        self.zrms = []
        self.t_rms = []
        
        self.U = Vector_field(params)
        self.set_arrays()
        # self.set_init(set_init)
        evolution.set_scheme(self) 
        
    def set_arrays(self):
        """Setup numpy/cupy arrays for wfc, wfck and pot for the simulation.
        """
        if self.params.dim == 1:
            self.wfc = self.params.xp.zeros((self.params.Nx), dtype = self.params.complex_dtype)
            self.wfck = self.params.xp.zeros((self.params.Nx), dtype = self.params.complex_dtype)
            self.pot = self.params.xp.zeros((self.params.Nx), dtype = self.params.real_dtype)
            
        elif self.params.dim == 2:
            self.wfc = self.params.xp.zeros((self.params.Nx, self.params.Ny), dtype = self.params.complex_dtype)
            self.wfck = self.params.xp.zeros((self.params.Nx, self.params.Ny), dtype = self.params.complex_dtype)
            self.pot = self.params.xp.zeros((self.params.Nx, self.params.Ny), dtype = self.params.real_dtype)
        
        else:
            self.wfc = self.params.xp.zeros((self.params.Nx, self.params.Ny, self.params.Nz), dtype = self.params.complex_dtype)
            self.pot = self.params.xp.zeros((self.params.Nx, self.params.Ny, self.params.Nz), dtype = self.params.real_dtype)
    
    def set_init(self, wfc_function, pot_function): #C
        """Initial setup of wavefunction and potential for the simulation.

        Parameters
        ----------
        fun : function
            returns the functional form of the wavefunction and potential.
        """
        if self.params.dim == 1:
            self.wfc[:], self.pot[:] = wfc_function(self.grid.xx), pot_function(self.grid.xx)
        
        if self.params.dim == 2:
            self.wfc[:], self.pot[:] = wfc_function(self.grid.xx, self.grid.yy), pot_function(self.grid.xx, self.grid.yy)
        
        if self.params.dim == 3:
            self.wfc[:], self.pot[:] = wfc_function(self.grid.xx, self.grid.yy, self.grid.zz), pot_function(self.grid.xx, self.grid.yy, self.grid.zz)
        # self.Npar = self.compute_norm()
    
    def compute_norm(self):  #C
        """Computes the normalisation constant for the current wavefunction.

        Returns
        -------
        float
            Normalisation constant
        """
        return fns.integralr(self.params, self.params.xp.abs(self.wfc)**2, self.grid)**0.5
    
    def renormalize(self, normFact: float = 1.0):   #C
        """Renormalisation of the wavefunction.

        Parameters
        ----------
        normFact : float, optional
            The new normalisation constant of the wavefunction, by default 1.0
        """
        self.wfc = normFact * self.wfc/self.compute_norm()
    
    def evolve(self):   
        evolution.time_advance(self)
    
    def evolve_ms(self):
        evolution.time_advance_ms(self)    
    
    def compute_chempot(self):   
        self.U.temp[:] = my_fft.forward_transform(self.params, self.wfc) #for sstep_strang
        deriv = self.params.volume * self.params.xp.sum(self.grid.ksqr * self.params.xp.abs(self.U.temp)**2)  
        return fns.integralr(self.params, ((self.pot + self.params.g * self.params.xp.abs(self.wfc)**2) * self.params.xp.abs(self.wfc)**2), self.grid) + deriv/2
    
    def compute_xrms(self):  #C
        return (fns.integralr(self.params, self.params.xp.abs(self.wfc)**2 * self.grid.xx**2, self.grid) - (fns.integralr(self.params, self.params.xp.abs(self.wfc)**2 * self.grid.xx, self.grid))**2)**.5
        
    def compute_yrms(self):  #C
        return (fns.integralr(self.params, self.params.xp.abs(self.wfc)**2 * self.grid.yy**2, self.grid) - (fns.integralr(self.params, self.params.xp.abs(self.wfc)**2 * self.grid.yy, self.grid))**2)**.5   

    def compute_zrms(self):  #C
        return (fns.integralr(self.params, self.params.xp.abs(self.wfc)**2 * self.grid.zz**2, self.grid) - (fns.integralr(self.params, self.params.xp.abs(self.wfc)**2 * self.grid.zz, self.grid))**2)**.5
    
    def compute_rrms(self):    #C
        if self.params.dim == 2:
            return (fns.integralr(self.params, self.params.xp.abs(self.wfc) ** 2 * (self.grid.xx**2 + self.grid.yy**2), self.grid))**.5
        return (fns.integralr(self.params, self.params.xp.abs(self.wfc) ** 2 * (self.grid.xx**2 + self.grid.yy**2 + self.grid.zz**2), self.grid))**.5
    
    def compute_energy(self):     #C
        self.U.temp[:] = my_fft.forward_transform(self.params, self.wfc) #for sstep_strang
        deriv = self.params.volume * self.params.xp.sum(self.grid.ksqr * self.params.xp.abs(self.U.temp)**2)  
        return fns.integralr(self.params, ((self.pot + 0.5 * self.params.g * self.params.xp.abs(self.wfc)**2) * self.params.xp.abs(self.wfc)**2), self.grid) + deriv/2
    
    def compute_quantum_energy(self):   #C
        fns.gradient(self.params, self.params.xp.abs(self.wfc), self)
        if self.params.dim == 2:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2) 

        elif self.params.dim == 3:
            self.U.temp[:] = 0.5 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        return fns.integralr(self.params, self.U.temp.real, self.grid)


    def compute_internal_energy(self):   
        return 0.5 * self.params.g * fns.integralr(self.params, self.params.xp.abs(self.wfc)**4, self.grid)
        

    def compute_potential_energy(self): 
        return fns.integralr(self.params, self.pot * self.params.xp.abs(self.wfc)**2, self.grid) 
        
    def compute_velocity(self):   
        fns.gradient(self.params, self.wfc.conj(), self)
        if self.params.dim == 2:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/self.params.xp.abs(self.wfc)**2 
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/self.params.xp.abs(self.wfc)**2 
    
        elif self.params.dim == 3:
            self.U.Vx[:] = -(self.wfc * self.U.Vx).imag/self.params.xp.abs(self.wfc)**2 
            self.U.Vy[:] = -(self.wfc * self.U.Vy).imag/self.params.xp.abs(self.wfc)**2 
            self.U.Vz[:] = -(self.wfc * self.U.Vz).imag/self.params.xp.abs(self.wfc)**2 
        return 
    

    def compute_kinetic_energy(self):  # doubt while integrating in fourier space
        self.compute_velocity()
        self.U.temp[:] = 0.5 * self.params.xp.abs(self.wfc)**2 * (self.U.Vx**2 + self.U.Vy**2 + self.U.Vz**2)
        return fns.integralr(self.params, self.U.temp.real, self.grid)

    def omegak(self):   
        self.compute_velocity() 
        self.U.temp[:] = self.params.xp.abs(self.wfc)
        self.U.omegai_kx[:] = my_fft.forward_transform(self.params, self.U.temp * self.U.Vx)
        self.U.omegai_ky[:] = my_fft.forward_transform(self.params, self.U.temp * self.U.Vy)
        
        if self.params.dim == 2:
            self.grid.ksqr[0, 0] = 1
            self.U.temp[:] = (self.grid.kxx * self.U.omegai_kx + self.grid.kyy * self.U.omegai_ky)/self.grid.ksqr
            # Compressible part calculation
            self.U.Vx[:] = self.grid.kxx * self.U.temp
            self.U.Vy[:] = self.grid.kyy * self.U.temp       
            self.grid.ksqr[0, 0] = 0   
            
            #incompressible part calculation
            self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
            self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy
        
        elif self.params.dim == 3:
            self.U.omegai_kz[:] = my_fft.forward_transform(self.params, self.U.temp * self.U.Vz)
            self.grid.ksqr[0, 0, 0] = 1 
            self.U.temp[:] = (self.grid.kxx * self.U.omegai_kx + self.grid.kyy * self.U.omegai_ky + self.grid.kzz * self.U.omegai_kz)/self.grid.ksqr
            # Compressible part calculation
            self.U.Vx[:] = self.grid.kxx * self.U.temp
            self.U.Vy[:] = self.grid.kyy * self.U.temp
            self.U.Vz[:] = self.grid.kzz * self.U.temp
            self.grid.ksqr[0, 0, 0] = 0 
            self.U.omegai_kx[:] = self.U.omegai_kx - self.U.Vx
            self.U.omegai_ky[:] = self.U.omegai_ky - self.U.Vy
            self.U.omegai_kz[:] = self.U.omegai_kz - self.U.Vz
        return 


    def KE_decomp(self):   
        """
        This function calculates the kinetic energy decomposition
        
        Returns
        -------
        arrays
            arrays containing the kinetic energy decomposition
        """
        self.omegak()
        if self.params.dim == 2:
            KE_comp = 0.5 * fns.integralk(self.params, self.params.xp.abs(self.U.Vx)**2 + self.params.xp.abs(self.U.Vy)**2, self.params)
            KE_incomp = 0.5 * fns.integralk(self.params, self.params.xp.abs(self.U.omegai_kx)**2 + self.params.xp.abs(self.U.omegai_ky)**2, self.params)
        else:
            KE_comp = 0.5 * fns.integralk(self.params, self.params.xp.abs(self.U.Vx)**2 + self.params.xp.abs(self.U.Vy)**2 + self.params.xp.abs(self.U.Vz)**2, self.params)
            KE_incomp = 0.5 * fns.integralk(self.params, self.params.xp.abs(self.U.omegai_kx)**2 + self.params.xp.abs(self.U.omegai_ky)**2 + self.params.xp.abs(self.U.omegai_kz)**2, self.params)
        return KE_comp, KE_incomp 
        
    
    # For calculation of particle number flux
    # def compute_tk_particle_no(self):    
    #     self.U.temp[:] = my_fft.forward_transform(self.params, self.wfc)
    #     self.U.temp1[:] = my_fft.forward_transform(self.params, self.params.g * self.wfc * self.params.xp.abs(self.wfc)**2 + self.wfc * self.pot)
    #     temp = (self.U.temp1[:] * self.params.xp.conjugate(self.U.temp)).imag
    #     return self.binning(temp)
    
    # def comp_par_no_spectrum(self):
    #     self.U.temp[:] = self.params.xp.abs(my_fft.forward_transform(self.params, self.wfc))**2
    #     return self.bining(self.U.temp)
    
    def comp_KEcomp_spectrum(self):
        self.omegak()
        if self.params.dim == 2:
            KE_incompk = 0.5 * (self.params.xp.abs(self.U.omegai_kx)**2 + self.params.xp.abs(self.U.omegai_ky)**2) 
            KE_compk = 0.5 * (self.params.xp.abs(self.U.Vx)**2 + self.params.xp.abs(self.U.Vy)**2)
        elif self.params.dim == 3:
            KE_incompk = 0.5 * (self.params.xp.abs(self.U.omegai_kx)**2 + self.params.xp.abs(self.U.omegai_ky)**2 + self.params.xp.abs(self.U.omegai_kz)**2) 
            KE_compk = 0.5 * (self.params.xp.abs(self.U.Vx)**2 + self.params.xp.abs(self.U.Vy)**2 + self.params.xp.abs(self.U.Vz)**2)

        KEcomp_spectrum = self.binning(KE_compk)
        KEincomp_spectrum = self.binning(KE_incompk)
        # print (KE_compk[2], KE_incompk[2])
        return KEcomp_spectrum, KEincomp_spectrum
    
    
    
    def binning(self, quantity):
        quantity_s = self.params.xp.zeros(self.params.Nx//2)
        for i in range(self.params.Nx//2):
            z = self.params.xp.where((self.grid.ksqr**.5 >= self.grid.kxx[i]) & (self.grid.ksqr**.5 < self.grid.kxx[i+1]))        
            quantity_s[i] = self.params.xp.sum(quantity[z])
        return quantity_s
