import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import xarray as xr

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Change directory to where to raw, un interpolated data is stored
os.chdir("/Volumes/MATHAM4/Sensitivity_tests")

# Base filename
fn = "Apollinaris_Patera_diagfi_volc_filt.nccomp.nc"

# Sensitivity test strings to preappend to fn
sensitivity_tests = ['high_H2O','high_MER','high_rho','med_rho','high_temp','low_temp','base']

# Set up figure
fig,ax = plt.subplots(4,2,sharex=True,sharey=True)
ax = ax.flatten()
volc_surf_dict = {}
max_vals = []
min_vals = []

# Read in sensitivity test output into dict. 
for test in sensitivity_tests:
    temp = xr.open_dataset(test + "_" + fn, mask_and_scale=False,decode_times=False)
    lat = temp.latitude
    lon = temp.longitude
    volc_surf_dict[test] = temp['volc_1_surf'][-1,:,:]
    
    
    max_vals.append(volc_surf_dict[test].max())
    min_vals.append(volc_surf_dict[test].min())

for idx,test in enumerate(sensitivity_tests):
    im = ax[idx].pcolormesh(lon,lat,volc_surf_dict[test],norm=colors.LogNorm(),edgecolors='none')
    ax[idx].set_title(test)
    ax[idx].grid(True)
    im.set_clim(min(min_vals),max(max_vals))


fig.colorbar(im,ax=ax.ravel().tolist(),label='Ash Area Density (kg/m^2)')
fig.suptitle('Apollinaris Patera Sensitivity Tests')
fig.supxlabel('Longitude')
fig.supylabel('Latitude')