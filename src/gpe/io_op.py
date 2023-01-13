import h5py as hp
from pathlib import Path
import shutil 
import os

from gpe.set_device import xp
from gpe.univ import fns


# directory generation
def gen_path():
    if not path.exists():
        os.makedirs(path)

    if not (path/'wfc').exists():   
        os.mkdir(path/'wfc')

# Function for input
def set_initcond(G):
    file_name = Path(G.params.in_path)/G.params.filename
    f = hp.File(file_name, 'r')
    if G.params.device == 'cpu':
        G.wfc = f['wfc']
        G.V = f['V']
    elif G.params.device == 'gpu':
        G.wfc = xp.asarray(f['wfc']) 
        G.V = xp.asarray(f['V'])
    f.close()
    
    
# Function to store the wavefunction   
def save_wfc(G, t):
    filename = path/('wfc/' + 'wfc_t%f.hdf5' %t)
    f = hp.f = hp.File(filename, 'w')
    if para.device == 'cpu':
        f.create_dataset('wfc', data = G.wfc)
    elif para.device == 'gpu':
        f.create_dataset('wfc', data = ncp.asnumpy(G.wfc))
    f.create_dataset('t', data = t) 
    f.close()


def compute_energy(G, t):
    norm = G.norm()
    comp_KE, incomp_KE = G.KE_decomp()
    if para.device == 'gpu':
        G.comp_KE.append(ncp.asnumpy(comp_KE/norm))
        G.incomp_KE.append(ncp.asnumpy(incomp_KE/norm))
        G.internal_energy.append(ncp.asnumpy(G.comp_internal_energy()/norm))
        G.quantum_energy.append(ncp.asnumpy(G.comp_quantum_energy()/norm))
        G.total_energy.append(ncp.asnumpy(G.comp_energy()/norm))
        G.total_KE.append(ncp.asnumpy(G.comp_kinetic_energy()/norm))
        G.potential_energy.append(ncp.asnumpy(G.comp_potential_energy()/norm))
        G.t_energy.append(t)
    
    elif para.device == 'cpu':
        G.comp_KE.append(comp_KE/norm)
        G.incomp_KE.append(incomp_KE/norm)
        G.internal_energy.append(G.comp_internal_energy()/norm)
        G.quantum_energy.append(G.comp_quantum_energy()/norm)
        G.total_energy.append(G.comp_energy()/norm)
        G.total_KE.append(G.comp_kinetic_energy()/norm)
        G.potential_energy.append(G.comp_potential_energy()/norm)
        G.t_energy.append(t)
    print(t, para.dt, G.total_energy[-1])
    
def compute_ektk(G, t):
    KEcomp_spec, KEincomp_spec = G.comp_KEcomp_spectrum()
    if para.device == 'gpu':
        G.KEcomp_spec.append(ncp.asnumpy(KEcomp_spec))
        G.KEincomp_spec.append(ncp.asnumpy(KEincomp_spec))
        G.tk_par_no.append(ncp.asnumpy(G.comp_tk_particle_no()))
        G.t_ektk.append(ncp.asnumpy(t))
    elif para.device == 'cpu':
        G.KEcomp_spec.append(KEcomp_spec)
        G.KEincomp_spec.append(KEincomp_spec)
        G.tk_par_no.append(G.comp_tk_particle_no())
        G.t_ektk.append(t)

def compute_rms(G, t):
    if para.device == 'cpu':
        G.xrms.append(G.comp_xrms())
        G.yrms.append(G.comp_yrms())
        if para.dimension == 3:
            G.zrms.append(G.comp_zrms())
        G.t_rms.append(t)
    
    elif para.device == 'gpu':
        G.xrms.append(ncp.asnumpy(G.comp_xrms()))
        G.yrms.append(ncp.asnumpy(G.comp_yrms()))
        if para.dimension == 3:
            G.zrms.append(ncp.asnumpy(G.comp_zrms()))
        G.t_rms.append(ncp.asnumpy(t))
    
def save_ektk(G):
    filename = path/'ektk.hdf5'
    f = hp.f = hp.File(filename, 'w')
    f.create_dataset('KEcomp_spec', data = G.KEcomp_spec)
    f.create_dataset('KEincomp_spec', data = G.KEincomp_spec)
    f.create_dataset('tk_par_no', data = G.tk_par_no)
    f.create_dataset('t_ektk', data = G.t_ektk)
        
def save_rms(G):
    filename = path/'rms.hdf5'
    f = hp.f = hp.File(filename, 'w')
    f.create_dataset('xrms', data = G.xrms)
    f.create_dataset('yrms', data = G.yrms)
    if para.dimension == 3:
        f.create_dataset('zrms', data = G.zrms)
    f.create_dataset('t_rms', data = G.t_rms)
    f.close()    

# Function to store the energy
def save_energy(G):
    filename = path/'energies.h5'
    f = hp.File(filename, 'w')
    f.create_dataset('Comp_KE', data = G.comp_KE)
    f.create_dataset('Incomp_KE', data = G.incomp_KE)
    f.create_dataset('Internal_Energy', data = G.internal_energy)
    f.create_dataset('Quantum_Energy', data = G.quantum_energy)
    f.create_dataset('Total_Energy', data = G.total_energy)
    f.create_dataset('Total_KE', data = G.total_KE)
    f.create_dataset('Potential_Energy', data = G.potential_energy)
    f.create_dataset('t', data = G.t_energy)
    f.close()

'''   
def save_para():
    file = Path(__file__).parent.resolve()
    shutil.copy(file/'para.py', path)

def print_params():
    fns.print_params()
'''