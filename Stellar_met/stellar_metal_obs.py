import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

from scipy.optimize import curve_fit


plt.rcParams.update({"text.usetex": True, "font.family": "serif", 'font.size': 13})
plt.rc('figure', facecolor='w')

'''

https://arxiv.org/abs/1704.00006
Fig 5: webplotdigitizer 


log(Z) = A log(Mstar) + B, and B = B(z) = b1 x z + b2 x z^2

 log(Z) = A log(Mstar) + B, and B = B(z) = b1 x z + b2 x z^2 (edited) 

 OR B(z)  or B(1+z) 

'''
### read data ###

#all_data = np.genfromtxt('stellar_obs.csv', delimiter=',', skip_header = 2)
all_data = np.genfromtxt('stellar_obs_all.csv', delimiter=',', skip_header = 2)
print(all_data.shape)

z_arr = [0.0, 0.7, 3.0]


mstar_all = []
z_all = []
metal_all = []
metal_up_all = []
metal_down_all = []

for z_ind in [0, 1, 2]:

    mstar_ind = all_data[::3, 2*z_ind]
    metal_ind = all_data[::3, 2*z_ind + 1]

    metal_up_ind = all_data[1::3, 2*z_ind + 1]
    metal_down_ind = all_data[2::3, 2*z_ind + 1]
    

    z_ind_arr = z_arr[z_ind]*np.ones_like(mstar_ind)


    mstar_all = np.append(mstar_all, mstar_ind)
    metal_all = np.append(metal_all, metal_ind)
    metal_up_all = np.append(metal_up_all, metal_up_ind)
    metal_down_all = np.append(metal_down_all, metal_down_ind)
    z_all = np.append(z_all, z_ind_arr)




print(mstar_all.shape, metal_all.shape, z_all.shape)

nan_ind = np.isnan(mstar_all)

mstar_all = mstar_all[~nan_ind]
metal_all = metal_all[~nan_ind]
metal_up_all = metal_up_all[~nan_ind]
metal_down_all = metal_down_all[~nan_ind]
z_all = z_all[~nan_ind]

print(mstar_all.shape, metal_all.shape, metal_up_all.shape, metal_down_all, z_all.shape)



######################################

def func(x, a, b, c):
    return a*np.sin(x[0])+b*np.cos(x[1])+c


def func(x, a, b, c):
    '''
    x[0] = log(Mstar)
    x[1] = redshift

    returns log(Metallicity)

    '''

    #logZ = a*x[0] + b*(x[1]) + c*(x[1])**2
    logZ = a*x[0] + b*(x[1]) + c
    #logZ = a*x[0] + b*(x[1] + 1) + c*((x[1] + 1)**2)
    return logZ



### fitting ####

'''
#limits = [0, 2*np.pi, 0, 2*np.pi]  # [x1_min, x1_max, x2_min, x2_max]
side_x = mstar_all # np.linspace(limits[0], limits[1], 100)
side_y = z_all # np.linspace(limits[2], limits[3], 10)
X1, X2 = np.meshgrid(side_x, side_y)
size = X1.shape
x1_1d = X1.reshape((1, np.prod(size)))
x2_1d = X2.reshape((1, np.prod(size)))


xdata = np.vstack((x1_1d, x2_1d))
'''

xdata = np.vstack((mstar_all, z_all))

#original = (3, 1, 0.5)
#z = func(xdata, *original)
#Z = z.reshape(size)
#z_noise = z + .2*np.random.randn(len(z))
#Z_noise = z_noise.reshape(size)

#ydata = z_noise

ydata = metal_all
yerr = np.vstack((metal_up_all, metal_down_all))
####

'''


limits = [0, 2*np.pi, 0, 2*np.pi]  # [x1_min, x1_max, x2_min, x2_max]
side_x = np.linspace(limits[0], limits[1], 100)
side_y = np.linspace(limits[2], limits[3], 10)
X1, X2 = np.meshgrid(side_x, side_y)
size = X1.shape
x1_1d = X1.reshape((1, np.prod(size)))
x2_1d = X2.reshape((1, np.prod(size)))


xdata = np.vstack((x1_1d, x2_1d))
original = (3, 1, 0.5)
z = func(xdata, *original)
Z = z.reshape(size)
z_noise = z + .2*np.random.randn(len(z))
Z_noise = z_noise.reshape(size)

ydata = z_noise
'''


print(xdata.shape, ydata.shape)

popt, pcov = curve_fit(func, xdata, ydata)
print("fitted a, b, c: {}".format(popt))
z_fit = func(xdata, *popt)
#Z_fit = z_fit.reshape(size)

'''
import matplotlib.pyplot as plt
plt.subplot(1, 3, 1)
plt.title("Real Function")
plt.pcolormesh(X1, X2, Z)
plt.axis(limits)
plt.colorbar()
plt.subplot(1, 3, 2)
plt.title("Function w/ Noise")
plt.pcolormesh(X1, X2, Z_noise)
plt.axis(limits)
plt.colorbar()
plt.subplot(1, 3, 3)
plt.title("Fitted Function from Noisy One")
plt.pcolormesh(X1, X2, Z_fit)
plt.axis(limits)
plt.colorbar()

plt.show()


'''


plt.figure(432, figsize = (7, 5))
color_arr = ['k', 'g', 'r']
for z_ind in [0, 1, 2]:

    data_idx = np.where(xdata[1] == z_arr[z_ind])
    #plt.scatter(xdata[0, data_idx], ydata[data_idx], marker = 'o', s = 25, label = 'Observed z=%.1f'%z_arr[z_ind])
    


    yerr_bars = np.array(list(zip( ydata[data_idx] - yerr[:, data_idx][1, 0, :] , yerr[:, data_idx][0, 0, :] - ydata[data_idx] ))).T

    #plt.errorbar(xdata[0, data_idx][0, :], ydata[data_idx], yerr = yerr[:, data_idx][:, 0, :] - ydata[data_idx] )
    plt.errorbar(xdata[0, data_idx][0, :], ydata[data_idx], color = color_arr[z_ind], yerr = yerr_bars, fmt = 'o',  label = 'Observed z=%.1f'%z_arr[z_ind])

    plt.text(9.0, 0.5 , r'$  \log_{10}(Z/Z_\odot) = A\log_{10}({\rm M_*}) + Bz + C $')
    xdata0_line = np.linspace(8.8, 11.0, 10)
    xdata1_line = z_arr[z_ind]*np.ones_like(xdata0_line)
    xdata_line = np.vstack((xdata0_line, xdata1_line))
    ydata_line = func(xdata_line, *popt)

    plt.plot(xdata_line[0], ydata_line, label = 'fit z=%.1f'%z_arr[z_ind], color = color_arr[z_ind], alpha = 0.8, linestyle='dashdot')




plt.ylim(-1.5, 0.7)
plt.xlim(8.8, 11.0)

plt.ylabel(r'$ {\rm Stellar } \log_{10}(Z/Z_\odot)$')
plt.xlabel(r'$\log_{10}({\rm M_*})[M_\odot]$')

plt.legend(ncol=2, loc = 'lower right')
plt.tight_layout()
plt.show()

print(10*'=')

#######################################
raise()

z_arr = [0.0, 0.7, 3.0]
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
