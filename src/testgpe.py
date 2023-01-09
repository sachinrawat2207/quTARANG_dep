from gpe.univ.grid import Grid
from gpe.univ.params import Params
from numpy import pi

par = Params(N = [10, 10, 20], L = [10, 10, 10], g = 1)
grid1 = Grid(par)

print(grid1.x, grid1.y, grid1.z)

# from gpe.wfc import wfc
# a = wfc()
print(grid1.dx, par.Lx/par.Nx, grid1.dkx, 2*pi/par.Lx)
print(grid1.dy, par.Ly/par.Ny, grid1.dky, 2*pi/par.Ly)
print(grid1.dz, par.Lz/par.Nz, grid1.dkz, 2*pi/par.Lz)