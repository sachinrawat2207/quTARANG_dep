import h5py as hp
import os
# from gpe.set_device import xp

# directory generation
def gen_path(file_loc):
    global path
    path = file_loc
    if not path.exists():
        os.makedirs(path)

    if not (path/'wfc').exists():   
        os.mkdir(path/'wfc')
    
    
# # Function for input
# def set_initcond(G):
#     file_name = Path(G.params.in_path)/G.params.filename
#     f = hp.File(file_name, 'r')
#     if G.params.gpu == True:
#         G.wfc = G.params.xp.asarray(f['wfc']) 
#         G.V = G.params.xp.asarray(f['V'])
    
#     else:
#         G.wfc = f['wfc']
#         G.V = f['V']
#     f.close()
    
    
def save_wfc(G, t):
    filename = path/('wfc/' + 'wfc_t%f.hdf5' %t)
    f = hp.f = hp.File(filename, 'w')
    if G.params.gpu == True:
        f.create_dataset('wfc', data = G.params.xp.asnumpy(G.wfc))
    else:
        f.create_dataset('wfc', data = G.wfc)  
    f.create_dataset('t', data = t) 
    f.close()


def compute_energy(G, t):
    norm = G.compute_norm()**2
    # comp_KE, incomp_KE = G.KE_decomp()
    if G.params.gpu == True:
        # G.comp_KE.append(G.params.xp.asnumpy(comp_KE/norm))
        # G.incomp_KE.append(G.params.xp.asnumpy(incomp_KE/norm))
        # G.internal_energy.append(G.params.xp.asnumpy(G.compute_internal_energy()/norm))
        # G.quantum_energy.append(G.params.xp.asnumpy(G.compute_quantum_energy()/norm))
        G.total_energy.append(G.params.xp.asnumpy(G.compute_energy()/norm))
        # G.total_KE.append(G.params.xp.asnumpy(G.compute_kinetic_energy()/norm))
        # G.potential_energy.append(G.params.xp.asnumpy(G.compute_potential_energy()/norm))
        G.t_energy.append(t)
    
    else:
        # G.comp_KE.append(comp_KE/norm)
        # G.incomp_KE.append(incomp_KE/norm)
        # G.internal_energy.append(G.compute_internal_energy()/norm)
        # G.quantum_energy.append(G.compute_quantum_energy()/norm)
        G.total_energy.append(G.compute_energy()/norm)
        # G.total_KE.append(G.compute_kinetic_energy()/norm)
        # G.potential_energy.append(G.compute_potential_energy()/norm)
        G.t_energy.append(t)
    # print(t, G.params.dt, G.total_energy[-1])
    
def compute_ektk(G, t):
    KEcomp_spec, KEincomp_spec = G.comp_KEcomp_spectrum()
    if G.params.gpu == True:
        G.KEcomp_spec.append(G.params.xp.asnumpy(KEcomp_spec))
        G.KEincomp_spec.append(G.params.xp.asnumpy(KEincomp_spec))
        # G.tk_par_no.append(G.params.xp.asnumpy(G.comp_tk_particle_no()))
        G.t_ektk.append(G.params.xp.asnumpy(t))
    else:
        G.KEcomp_spec.append(KEcomp_spec)
        G.KEincomp_spec.append(KEincomp_spec)
        # G.tk_par_no.append(G.comp_tk_particle_no())
        G.t_ektk.append(t)

def compute_rms(G, t):
    if G.params.gpu == True:
        G.xrms.append(G.params.xp.asnumpy(G.compute_xrms()))
        G.yrms.append(G.params.xp.asnumpy(G.compute_yrms()))
        if G.params.dim == 3:
            G.zrms.append(G.params.xp.asnumpy(G.compute_zrms()))
        G.t_rms.append(G.params.xp.asnumpy(t))
    
    else:
        G.xrms.append(G.compute_xrms())
        G.yrms.append(G.compute_yrms())
        if G.params.dim == 3:
            G.zrms.append(G.compute_zrms())
        G.t_rms.append(t)
    
def save_ektk(G):
    filename = path/'ektk.hdf5'
    f = hp.f = hp.File(filename, 'w')
    f.create_dataset('KEcomp_spec', data = G.KEcomp_spec)
    f.create_dataset('KEincomp_spec', data = G.KEincomp_spec)
    # f.create_dataset('tk_par_no', data = G.tk_par_no)
    # f.create_dataset('t_ektk', data = G.t_ektk)
        
def save_rms(G):
    filename = path/'rms.hdf5'
    f = hp.f = hp.File(filename, 'w')
    f.create_dataset('xrms', data = G.xrms)
    f.create_dataset('yrms', data = G.yrms)
    if G.params.dim == 3:
        f.create_dataset('zrms', data = G.zrms)
    f.create_dataset('t_rms', data = G.t_rms)
    f.close()    

# Function to store the energy
def save_energy(G):
    filename = path/'energies.h5'
    f = hp.File(filename, 'w')
    # f.create_dataset('Comp_KE', data = G.comp_KE)
    # f.create_dataset('Incomp_KE', data = G.incomp_KE)
    # f.create_dataset('Internal_Energy', data = G.internal_energy)
    # f.create_dataset('Quantum_Energy', data = G.quantum_energy)
    f.create_dataset('Total_Energy', data = G.total_energy)
    # f.create_dataset('Total_KE', data = G.total_KE)
    # f.create_dataset('Potential_Energy', data = G.potential_energy)
    f.create_dataset('t', data = G.t_energy)
    f.close()
