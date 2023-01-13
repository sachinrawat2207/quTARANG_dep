from gpe.univ import my_fft
from gpe.set_device import xp
import gpe.in_op as in_op
from pathlib import Path
#-----------------------------------------TSSP scheme-----------------------------------------
def tssp_stepr(G, dt: float):
    return G.wfc * xp.exp(-1j * (G.pot + G.params.g  * (G.wfc * G.wfc.conj())) * dt)

def tssp_stepk(G, dt: float):
    return G.wfck * xp.exp(-0.5j * G.grid.ksqr * dt)

# For real time evolution
def time_adv_strang(G):
    G.wfc = tssp_stepr(G, G.params.dt/2)
    G.wfck = my_fft.forward_transform(G.wfc)
    G.wfck = tssp_stepk(G, G.params.dt)
    G.wfc = my_fft.inverse_transform(G.wfck)
    G.wfc = tssp_stepr(G, G.params.dt/2)

# For imaginary time evolution
def time_adv_istrang(G):
    G.wfc = tssp_stepr(G, -1j * G.params.dt/2)
    G.wfck = my_fft.forward_transform(G.wfc)
    G.wfck = tssp_stepk(G, -1j * G.params.dt)
    G.wfc = my_fft.inverse_transform(G.wfck)
    G.wfc = tssp_stepr(G, -1j * G.params.dt/2)
    G.renormalize(G.Npar)
    # G.wfc = G.wfc/(G.params.volume * xp.sum(xp.abs(my_fft.forward_transform(G.wfc))**2))**.5


#-----------------------------------------RK4 scheme-----------------------------------------
def compute_RHS(G, psik):
    psi = my_fft.inverse_transform(psik)
    psi = -1j  * (my_fft.ksqr * psik/2 + my_fft.forward_transform((G.params.g * xp.abs(psi)**2 + G.V) * psi))
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
    in_op.gen_path(Path.cwd()) 
    
    for i in range(G.params.nstep):

        if(G.params.save_wfc == True and i >= G.params.save_wfc_start_step and (i - G.params.save_wfc_start_step)%G.params.save_wfc_iter_step == 0):
            in_op.save_wfc(G, t)
            
        if(G.params.save_energy == True and i >= G.params.save_en_start_step and (i - G.params.save_en_start_step)%G.params.save_en_iter_step == 0):
            in_op.compute_energy(G, t)
        
        if(G.params.save_ektk == True and i >= G.params.save_ektk_start_step and (i - G.params.save_ektk_start_step)%G.params.save_ektk_iter_step == 0):
            in_op.compute_ektk(G, t)

        if(G.params.save_rms == True and i >= G.params.save_rms_start_step and  (i - G.params.save_rms_start_step)%G.params.save_rms_iter_step == 0):
            G.xrms.append(G.compute_xrms())
            G.yrms.append(G.compute_yrms())
            G.zrms.append(G.compute_zrms())
            G.t_rms.append(t)
        
        t += G.params.dt
        time_adv(G)

    if (G.params.save_energy): 
        in_op.compute_energy(G, t)
        in_op.save_energy(G)
    if G.params.save_wfc == True:
        in_op.save_wfc(G, t)
        
    if G.params.save_rms == True:
        in_op.compute_rms(G, t)
        in_op.save_rms(G)
    
    if G.params.save_ektk == True:
        in_op.compute_ektk(G, t)
        in_op.save_ektk(G)
