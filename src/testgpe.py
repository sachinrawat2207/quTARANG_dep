from gpe.univ.grid import Grid
from gpe.univ.params import Params
from gpe.gpe import GPE
from gpe.set_device import xp
from numpy import pi

def set_init(x, y=0):
    wfc = (1/xp.pi)**0.5 * xp.exp(-(x**2 +y**2)/2)
    pot = 0.5*(x**2 + y**2) 
    return wfc, pot
   

# def set_init(x, y=0, z=0):
#     wfc = (8**0.25)*(0.25*xp.pi)**(-3/8) * xp.exp(2*(-x**2 - 2*y**2 - 4*z**2))
#     V = 1/2*(x**2 + 4*y**2 * 16*z**2)
#     return wfc, V

par = Params(N = [32, 32, 1], L = [16, 16, 1], g = 0.1, dt=0.001)
gpe = GPE(par, set_init)
gpe.evolve()

# print(type(gpe.wfc), gpe.wfc.shape)
