import h5py as hp
import numpy as np
import matplotlib.pyplot as plt

file_name = 'rms.hdf5'
f = hp.File(file_name, 'r')
data_time_interval = np.array(f['Time'])
data_obtained_xrms = np.array(f['x_rms'])
data_obtained_yrms = np.array(f['y_rms'])
data_obtained_zrms = np.array(f['z_rms'])
f.close()

## data from reference
xfile1, yfile1 = np.loadtxt('ref_data/3D_I_a.txt', delimiter = ',', unpack=True)
xfile2, yfile2 = np.loadtxt('ref_data/3D_I_b.txt', delimiter = ',', unpack=True)
xfile3, yfile3 = np.loadtxt('ref_data/3D_I_c.txt', delimiter = ',', unpack=True)


## Plot Settings
# matplotlib.style.use('classic')
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 1
# plt.rcParams['xtick.minor.size'] = 8
# plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 1
# plt.rcParams['ytick.minor.size'] = 5
# plt.rcParams['ytick.minor.width'] = 1

plt.rc('text', usetex=True)
# plt.rc('axes', linewidth=1.5)
plt.rc('font', weight='bold')
# font = {'family' : 'serif', 'weight' : 'bold', 'size' : 30}
# plt.rc('font', **font)



## Plotting obtained data with reference data for comparison.
plt.rcParams.update({'font.size':'14'})
plt.figure(figsize=(6, 5))
# plt.title("3D Dynamics")

plt.ylabel("$\sigma$", fontsize = 15)
plt.xlabel(r"$t$", fontsize = 15)
plt.xlim(0, data_time_interval[-1])
plt.yticks([0, 0.4, 0.8, 1.2, 1.6])
plt.ylim(0, 1.6)
plt.plot(data_time_interval, data_obtained_xrms,'k:', label='$\sigma_x$')#, color='k', linewidth=2)
plt.plot(data_time_interval, data_obtained_yrms,'k-.', label='$\sigma_y$')#, color = 'red', linewidth=2)
plt.plot(data_time_interval, data_obtained_zrms,'k-', label='$\sigma_z$')#, color='blue', linewidth=2)
plt.scatter(xfile1[::4], yfile1[::4], s=30, label="Bao($\sigma_x$)", marker='s', color = 'k')#, color = 'k', )
plt.scatter(xfile2[::5], yfile2[::5], s=30, label="Bao($\sigma_y$)", marker='^', color = 'k')#, color = 'red', marker='s')
plt.scatter(xfile3[::5], yfile3[::5], s=30, label="Bao($\sigma_z$)", marker='p', color = 'k')#, color='blue', marker='^')
plt.legend(loc=0, bbox_to_anchor=(0.63, 0.78), fontsize = 11, ncol = 2, frameon=False, columnspacing=0.8)
plt.savefig("dynamics.jpeg", dpi=300, bbox_inches='tight')
# plt.show()