import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

plt.rcParams.update({"text.usetex": True, "font.family": "serif", 'font.size': 14})
plt.rc('figure', facecolor='w')

'''

https://arxiv.org/abs/1704.00006
Fig 5: webplotdigitizer 

'''


all_data = np.genfromtxt('stellar_metal_z_mstar.csv', delimiter=',', skip_header = 2)
#print(all_data.shape)

z_arr = [0.0, 1.0, 2.0, 3.0]
mstar_arr_new = np.linspace(9, 11, 15)

X, Y = np.meshgrid(z_arr, mstar_arr_new)

metal_2d_input = np.zeros_like(X)
print(metal_2d_input.shape)

plt.figure(12, figsize = (7, 5))
for z_ind in [0, 1, 2, 3]:

    mstar_ind = all_data[:, 2*z_ind]
    metal_ind = all_data[:, 2*z_ind + 1]

    nan_ind = np.isnan(mstar_ind)
    mstar_ind = mstar_ind[~nan_ind]
    metal_ind = metal_ind[~nan_ind]
    
    print(mstar_ind.max(), metal_ind.max())
    plt.scatter(mstar_ind, metal_ind, marker = 'x', s = 15)

    metal_interp = interp1d(mstar_ind, metal_ind, fill_value="extrapolate")
    metal_new = metal_interp(mstar_arr_new)

    #plt.plot(mstar_arr_new, metal_new, linestyle = 'dashdot', alpha = 0.8, linewidth = 2)
    metal_2d_input[:, z_ind] = metal_new

print(metal_2d_input)

metal_interp2d = interp2d(z_arr, mstar_arr_new, metal_2d_input, kind='linear')


z_new_num = 7
z_arr_new = np.linspace(0.0, 3.0, z_new_num)
metal_2d_new = metal_interp2d(z_arr_new, mstar_arr_new)


for z_new in z_arr_new:
    plt.plot( mstar_arr_new, metal_interp2d(z_new, mstar_arr_new), label = 'z = %0.1f'%z_new, linestyle = 'dashdot', alpha = 0.5, linewidth = 1.5 )


plt.ylabel(r'$ {\rm Stellar } \log_{10}(Z/Z_\odot)$')
plt.xlabel(r'$\log_{10}({\rm M_*})[M_\odot]$')

plt.legend(ncol=2)
plt.tight_layout()
plt.show()
