from gpe.univ.grid import Grid
from gpe.univ.params import Params
from gpe.set_device import xp
from gpe.univ import my_fft
from gpe import gpe

N = [128, 128, 1]
L = [29, 29, 1]
ektk = [True, 0, 10000000]
wfc = [True, 0, 100]

par = Params(N, L, g = 0.1, dt=0.001, tmax = 2, energy = [True, 0, 100], ektk = ektk, wfc = wfc, path = '/home/Desktop/test')
G = gpe.GPE(par)

def set_init(x, y=0):
    wfc = (1/xp.pi)**0.5 * xp.exp(-(x**2 +y**2)/2)
    pot = 0.5*(x**2 + y**2) 
    return wfc, pot
    
def set_init_tsubota(x, y=0, z=0):
    xp.random.seed(0)
    theta0 = 1
    dk = 2*xp.pi/L[0]
    kf1 = dk
    kf2 = 3*dk
    ## alpha(k) and theta(k) for whole k space
    alpha_k = 0.9999*xp.pi * (xp.random.random((N[0], N[1]))*2 - 1)
    theta_k = theta0 * xp.exp(1j*alpha_k)
    
    ## cuting out the portion between dk and 3dk
    k_abs = xp.sqrt(G.grid.ksqr)
    filter_plate = (k_abs >= kf1) & (k_abs <= kf2)
    filter_plate = filter_plate*1
    # filter_plate = xp.int32(filter_plate)
    theta_k[xp.where(xp.logical_not(filter_plate))] = 0
    
    ## satisfying symmetry condition
    mid = N[0]//2
    temp = xp.roll(theta_k, shift=(mid, mid), axis=(0,1))
    res = xp.zeros_like(temp)
    res[mid:, 1:mid+1] = temp[mid:, 1:mid+1]
    res[1:mid, 1:mid] = temp[1:mid, 1:mid]
    res[1:mid+1, mid:] = xp.conjugate(xp.flip(temp[mid:,1:mid+1], axis=(0,1)))
    res[mid+1:, mid+1:] = xp.conjugate(xp.flip(temp[1:mid,1:mid], axis=(0,1)))
    theta_k = xp.roll(res, shift=(mid, mid), axis=(0,1))    
    theta = my_fft.inverse_transform(theta_k)
    psi = xp.exp(1j * theta)
    return psi, 0
    
def set_init2D(x, y=0, z=0):
    wfc = (8**0.25)*(0.25*xp.pi)**(-3/8) * xp.exp(2*(-x**2 - 2*y**2 - 4*z**2))
    V = 1/2*(x**2 + 4*y**2 * 16*z**2)
    return wfc, V

print(par)

G.set_init(set_init_tsubota)

for i in range(10):
    G.evolve_ms()
    comp, incomp = G.KE_decomp()
    print(G.compute_energy()/G.compute_norm()**2, G.compute_kinetic_energy()/G.compute_norm()**2 + G.compute_quantum_energy()/G.compute_norm()**2 + G.compute_internal_energy()/G.compute_norm()**2)
    print(G.compute_kinetic_energy()/G.compute_norm()**2, comp/G.compute_norm()**2 + incomp/G.compute_norm()**2 )
    
print(type(G.wfc), G.wfc.shape)