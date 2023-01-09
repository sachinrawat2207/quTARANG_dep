import para
import my_fft
import in_op as in_op
from gpe.gpeset_device import xp

#-----------------------------------------TSSP scheme-----------------------------------------
def tssp_stepr(G, dt):
    return G.wfc * xp.exp(-1j * (G.V + para.g  * (G.wfc * G.wfc.conj())) * dt)

def tssp_stepk(G, dt):
    return G.wfck * xp.exp(-1j * (my_fft.ksqr/2) * dt)

# For real time evolution
def time_adv_strang(G):
    G.wfc = tssp_stepr(G, para.dt/2)
    G.wfck = my_fft.forward_transform(G.wfc)
    G.wfck = tssp_stepk(G, para.dt)
    G.wfc = my_fft.inverse_transform(G.wfck)
    G.wfc = tssp_stepr(G,para.dt/2)

# For imaginary time evolution
def time_adv_gstate_strang(G):
    G.wfc = tssp_stepr(G, -1j * para.dt/2)
    G.wfck = my_fft.forward_transform(G.wfc)
    G.wfck = tssp_stepk(G, -1j * para.dt)
    G.wfc = my_fft.inverse_transform(G.wfck)
    G.wfc = tssp_stepr(G, -1j * para.dt/2)
    G.wfc=G.wfc/(para.volume * xp.sum(xp.abs(my_fft.forward_transform(G.wfc))**2))**.5


#-----------------------------------------RK4 scheme-----------------------------------------
def compute_RHS(G,psik):
    psi = my_fft.inverse_transform(psik)
    psi = -1j  *(my_fft.ksqr * psik/2 + my_fft.forward_transform((para.g * xp.abs(psi)**2 + G.V) * psi))
    return psi

#-----------------------------------------Time Advance-----------------------------------------
def time_advance(G, U):
    if para.scheme == 'TSSP':
        if para.evolution == 'real':
            time_adv = time_adv_strang
        
        elif para.evolution == 'imag':
            time_adv = time_adv_gstate_strang
            
    else:
        print("Please choose the correct scheme")
        quit()
  
    t = 0 
    for i in range(para.nstep):
        if(i >= para.save_wfc_start_step and (i - para.save_wfc_start_step)%para.save_wfc_iter_step == 0):
            in_op.save_wfc(G, t)
            
        if(i >= para.save_en_start_step and (i - para.save_en_start_step)%para.save_en_iter_step == 0):
            in_op.compute_energy(G, U, t)
        
        if(para.save_ektk == True and i >= para.save_ektk_start_step and (i - para.save_ektk_start_step)%para.save_ektk_iter_step == 0):
            in_op.compute_ektk(G, U, t)
        
        if(para.save_rms == True and i >= para.save_en_start_step and  (i - para.save_en_start_step)%para.save_en_iter_step == 0):
            in_op.compute_rms(G, t)
        
        t = t + para.dt
        time_adv(G)

    in_op.save_wfc(G, t)
    in_op.compute_energy(G, U, t)
    in_op.save_energy(G)
    
    if para.save_rms == True:
        in_op.compute_rms(G, t)
        in_op.save_rms(G)
    
    if para.save_ektk == True:
        in_op.compute_ektk(G, U, t)
        in_op.save_ektk(G)
        