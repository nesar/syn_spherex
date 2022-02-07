y-axis: Flux per unit wavelength (ergs/s/cm^2/Angstrom)
x-axis: Restframe Wavelength (Angstrom)


#### DUST CONVERSION ####
dust_fac = 1/(0.4/0.44/np.log10(np.e))/2
ebv_val = dust_range*dust_fac
###########################

