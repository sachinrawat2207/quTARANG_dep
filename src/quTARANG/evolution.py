# from gpe.set_device import xp
from quTARANG.univ import my_fft
import quTARANG.IO as IO
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


#-----------------------------------------RK4 scheme-----------------------------------------
def compute_RHS(G, psik):
    psi = my_fft.inverse_transform(G.params, psik)
    psi = -1j  * (my_fft.ksqr * psik/2 + my_fft.forward_transform(G.params, (G.params.g * G.params.xp.abs(psi)**2 + G.V) * psi))
    return psi

#-----------------------------------------Time Advance-----------------------------------------

def set_scheme(G):
    global time_adv
    if G.params.scheme == 'TSSP':
        if G.params.imgtime == False:
            time_adv = time_adv_strang
        elif G.params.imgtime == True:
            time_adv = time_adv_istrang
    else:
        print("Please choose the correct scheme")
        quit()


def time_advance_ms(G):
    time_adv(G)
    
def time_advance(G):
    t = 0 
    
    if (G.params.save_wfc or G.params.save_energy or G.params.save_ektk or G.params.save_rms):
        IO.gen_path(G.params.path) 
    
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
