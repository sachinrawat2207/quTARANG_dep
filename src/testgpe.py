from gpe.univ.grid import Grid
from gpe.univ.params import Params
from gpe.univ.gpe import GPE
from numpy import pi

par = Params(N = [10, 10, 20], L = [10, 10, 10], g = 1)
gpe = GPE(par)

print(type(gpe.wfc), gpe.wfc.shape)