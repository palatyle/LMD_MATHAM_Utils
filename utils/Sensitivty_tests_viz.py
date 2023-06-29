import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import boxcox, pearsonr

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
os.chdir("/Volumes/MATHAM4/Sensitivity_tests")

filename_common = "Apollinaris_Patera_diagfi_volc_filt.nccomp.nc"

sensitivity_tests = ['high_H2O','high_MER','high_rho','med_rho','high_temp','low_temp','base']

fig,ax = plt.subplots(4,2,sharex=True,sharey=True)
ax = ax.flatten()
volc_surf_dict = {}
max_vals = []
min_vals = []
for test in sensitivity_tests:
    temp = xr.open_dataset(test+ "_" + filename_common, mask_and_scale=False,decode_times=False)
    lat = temp.latitude
    lon = temp.longitude
    volc_surf_dict[test] = temp['volc_1_surf'][-1,:,:]
    
    
    max_vals.append(volc_surf_dict[test].max())
    min_vals.append(volc_surf_dict[test].min())

print('stop')

for idx,test in enumerate(sensitivity_tests):
    im = ax[idx].pcolormesh(lon,lat,volc_surf_dict[test],norm=colors.LogNorm(),edgecolors='none')
    ax[idx].set_title(test)
    ax[idx].grid(True)
    im.set_clim(min(min_vals),max(max_vals))


fig.colorbar(im,ax=ax.ravel().tolist(),label='Ash Area Density (kg/m^2)')
fig.suptitle('Apollinaris Patera Sensitivity Tests')
fig.supxlabel('Longitude')
fig.supylabel('Latitude')

base_transformed = boxcox(volc_surf_dict['base'].stack(z=("longitude","latitude")))
for test in sensitivity_tests:
    transformed = boxcox(volc_surf_dict[test].stack(z=("longitude","latitude")))
    print(test)
    print(pearsonr(transformed[0],base_transformed[0]))