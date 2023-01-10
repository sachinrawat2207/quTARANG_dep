from gpe.univ.grid import Grid
from gpe.univ.params import Params
from gpe.evolution import time_adv_strang
from gpe.gpe import GPE
from gpe.set_device import xp
from numpy import pi
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_init(x, y=0, z=0):
    eps1 = 0.25
    normfactor = 2**0.25/(xp.pi*eps1)**(3/8)
    wfc = normfactor * xp.exp(-(x**2 + 2*y**2 + 4*z**2)/(2*eps1))
    pot = 0.5*(x**2 + 4*y**2 + 16*z**2)
    return wfc, pot

par = Params(
    N = [64, 64, 64],
    L = [16, 16, 16],
    g = 0.1,
    dt = 0.001)
gpe = GPE(par, set_init)
gpe.renormalize()

xrms_list = []
yrms_list = []
zrms_list = []
Tf = 20
Nt = int(Tf/par.dt)
t = xp.linspace(0, Tf, Nt)
for _ in tqdm(range(Nt)):
    time_adv_strang(gpe)
    xrms_list.append(gpe.compute_xrms())
    yrms_list.append(gpe.compute_yrms())
    zrms_list.append(gpe.compute_zrms())

xfile1, yfile1 = xp.loadtxt('src/bao_data/3D_I_a.txt', delimiter = ',', unpack=True)
xfile2, yfile2 = xp.loadtxt('src/bao_data/3D_I_b.txt', delimiter = ',', unpack=True)
xfile3, yfile3 = xp.loadtxt('src/bao_data/3D_I_c.txt', delimiter = ',', unpack=True)

xrms_list = xp.array(xrms_list).real
yrms_list = xp.array(yrms_list).real
zrms_list = xp.array(zrms_list).real

plt.rcParams.update({'font.size':20})
plt.figure(figsize=(7, 7))
plt.title("2D case I")
plt.xlim(t[0], t[-1])
plt.ylim(0.7, 0.82)
plt.plot(t, xrms_list, label='$\sigma_x$')
plt.plot(t, yrms_list, 'r--', dashes=(5,5), label='$\sigma_y$')
plt.scatter(xfile1, yfile1, s=20, label="Bao($\sigma_x$)")
plt.scatter(xfile2, yfile2, s=20, label="Bao($\sigma_y$)")
plt.scatter(xfile3, yfile3, s=20, label="Bao($\sigma_z$)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()