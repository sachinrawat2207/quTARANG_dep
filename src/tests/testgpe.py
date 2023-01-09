from gpe.univ.grid import Grid
from gpe.univ.params import Params

par = Params(N = [32, 34, 36], L = [10, 12, 14], g = 1)
grid1 = Grid(par)

print(grid1.x, grid1.y, grid1.z)

# from gpe.wfc import wfc
# a = wfc()