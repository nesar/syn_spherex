import matplotlib.pylab as plt
import numpy as np

sfhID = 642
dust_range = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

plt.figure(1)
for dust2val in dust_range:
    spec_file = './dust/spec_sfhID_'+str(sfhID) + '_dustEBV_'+str(dust2val) + '.txt'
    wave, spec = np.loadtxt(spec_file)
    plt.plot(wave, spec, label = 'E(B-V) = %.1f'%dust2val)


plt.ylabel(' Flux per unit wavelength (ergs/s/cm^2/Angstrom)')
plt.xlabel(' Restframe Wavelength (Angstrom) ')
plt.yscale('log')
plt.ylim(1e-35, 1e-11)
plt.legend()
plt.show()
