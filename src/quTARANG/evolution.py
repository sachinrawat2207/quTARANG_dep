# from gpe.set_device import xp
from quTARANG.univ import my_fft
import quTARANG.IO as IO
from quTARANG.univ import fns
from tqdm import tqdm

#-----------------------------------------TSSP scheme-----------------------------------------
def tssp_stepr(G, dt: float):
    """ Step of the TSSP Scheme to be taken in real space
    
    Parameters
    ----------
    G : GPE class
    dt : float
        
    
    Returns
    -------
    ndarray
        wavefunction after evolution
    """
    return G.wfc * G.params.xp.exp(-1j * (G.pot + G.params.g  * (G.wfc * G.wfc.conj())) * dt)

def tssp_stepk(G, dt: float):
    """ Step of the TSSP Scheme to be taken in k space
    
    Parameters
    ----------
    G : GPE class
    dt : float
       
    
    Returns
    -------
    ndarray
        wavefunction after evolution
    """
    return G.wfck * G.params.xp.exp(-0.5j * G.grid.ksqr * dt)

# For real time evolution
def time_adv_strang(G):
    G.wfc = tssp_stepr(G, G.params.dt/2)
    G.wfck = my_fft.forward_transform(G.params, G.wfc)
    G.wfck = tssp_stepk(G, G.params.dt)
    G.wfc = my_fft.inverse_transform(G.params, G.wfck)
    G.wfc = tssp_stepr(G, G.params.dt/2)

# For imaginary time evolution
def time_adv_istrang(G):
    G.wfc = tssp_stepr(G, -1j * G.params.dt/2)
    G.wfck = my_fft.forward_transform(G.params, G.wfc)
    G.wfck = tssp_stepk(G, -1j * G.params.dt)
    G.wfc = my_fft.inverse_transform(G.params, G.wfck)
    G.wfc = tssp_stepr(G, -1j * G.params.dt/2)
    G.renormalize(G.Npar)
    # G.wfc = G.wfc/(G.params.volume * xp.sum(xp.abs(my_fft.forward_transform(G.wfc))**2))**.5


#-----------------------------------------Compute RHS function-----------------------------------------
def compute_rhs(G, psik):
    psi = my_fft.inverse_transform(G.params, psik)
    psi = -1j  * (my_fft.ksqr * psik/2 + my_fft.forward_transform(G.params, (G.params.g * G.params.xp.abs(psi)**2 + G.V) * psi))
    return psi

#-----------------------------------------RK4 scheme-----------------------------------------

def  time_adv_rk4(G):
    """ RK4 scheme for evolution
    
    Parameters
    ----------
    G : GPE class
    """
    k1 = compute_rhs(G, G.wfck)
    k2 = compute_rhs(G, G.wfck + G.params.dt/2 * k1)
    k3 = compute_rhs(G, G.wfck + G.params.dt/2 * k2)
    k4 = compute_rhs(G, G.wfck + G.params.dt * k3)
    G.wfck = G.wfck + G.params.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    

def time_adv_irk4(G):
    """ RK4 scheme for imaginary time evolution
    
    Parameters
    ----------
    G : GPE class
    """
    k1 = compute_rhs(G, G.wfck)
    k2 = compute_rhs(G, G.wfck + 1j * G.params.dt/2 * k1)
    k3 = compute_rhs(G, G.wfck + 1j * G.params.dt/2 * k2)
    k4 = compute_rhs(G, G.wfck + 1j * G.params.dt * k3)
    G.wfck = G.wfck + 1j * G.params.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    G.wfck = G.wfck / (fns.integralk(G.params, G.params.xp.abs(G.wfck)**2))**0.5

#-----------------------------------------Time Advance-----------------------------------------

def set_scheme(G):
    global time_adv
    if G.params.scheme == 'TSSP':
        if G.params.imgtime == False:
            time_adv = time_adv_strang
        elif G.params.imgtime == True:
            time_adv = time_adv_istrang
    
    elif G.params.scheme == 'RK4':        
        if G.params.imgtime == False:
            time_adv = time_adv_rk4
        elif G.params.imgtime == True:
            time_adv = time_adv_irk4
    
    else:
        print("Please choose the correct scheme")
        quit()


def time_advance_ms(G):
    time_adv(G)
    
def time_advance(G):
    t = 0 
    
    if (G.params.save_wfc or G.params.save_energy or G.params.save_ektk or G.params.save_rms):
        IO.gen_path(G.params.path) 
    
    # For schemes ther then TSSP, where we need not to come in real space
    if G.params.scheme != 'TSSP':
        G.wfck = my_fft.forward_transform(G.wfc) 
        
    for i in tqdm(range(G.params.nstep)):
        if(G.params.save_wfc == True and i >= G.params.save_wfc_start_step and (i - G.params.save_wfc_start_step)%G.params.save_wfc_iter_step == 0):
            IO.save_wfc(G, t)
            
        if(G.params.save_energy == True and i >= G.params.save_en_start_step and (i - G.params.save_en_start_step)%G.params.save_en_iter_step == 0):
            IO.compute_energy(G, t)
        
        if(G.params.save_ektk == True and i >= G.params.save_ektk_start_step and (i - G.params.save_ektk_start_step)%G.params.save_ektk_iter_step == 0):
            IO.compute_ektk(G, t)

        if(G.params.save_rms == True and i >= G.params.save_rms_start_step and  (i - G.params.save_rms_start_step)%G.params.save_rms_iter_step == 0):
            IO.compute_rms(G, t)
        
        t += G.params.dt
        time_adv(G)
        
        if G.params.scheme != 'TSSP':
            G.wfc = my_fft.inverse_transform(G.wfck)
    
    
    if (G.params.save_energy): 
        IO.compute_energy(G, t)
        IO.save_energy(G)
        
    if G.params.save_wfc == True:
        IO.save_wfc(G, t)
        
    if G.params.save_rms == True:
        IO.compute_rms(G, t)
        IO.save_rms(G)
    
    if G.params.save_ektk == True:
        IO.compute_ektk(G, t)
        IO.save_ektk(G)
