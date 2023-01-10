from gpe.univ.grid import Grid
from gpe.univ.params import Params
from gpe.evolution import time_adv_strang
from gpe.gpe import GPE
from gpe.set_device import xp
from numpy import pi
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_init(x, y=0):
    wfc = (1/xp.pi)**0.5 * xp.exp(-(x**2 +y**2)/2)
    pot = 0.5*(x**2 + y**2) 
    return wfc, pot

par = Params(
    N = [128, 128, 1],
    L = [16, 16, 1],
    g = 2,
    dt = 0.001)
gpe = GPE(par, set_init)

xrms_list = []
yrms_list = []
Tf = 20
Nt = int(Tf/par.dt)
t = xp.linspace(0, Tf, Nt)
for _ in tqdm(range(Nt)):
    time_adv_strang(gpe)
    xrms_list.append(gpe.compute_xrms())
    yrms_list.append(gpe.compute_yrms())

xfile, yfile = xp.loadtxt('src/bao_data/2D_I.txt', delimiter = ',', unpack=True)

xrms_list = xp.array(xrms_list).real
yrms_list = xp.array(yrms_list).real

plt.rcParams.update({'font.size':20})
plt.figure(figsize=(7, 7))
plt.title("2D case I")
plt.xlim(t[0], t[-1])
plt.ylim(0.7, 0.82)
plt.plot(t, xrms_list, label='$\sigma_x$')
plt.plot(t, yrms_list, 'r--', dashes=(5,5), label='$\sigma_y$')
plt.scatter(xfile, yfile, s=20, label="Bao($\sigma_x$)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()