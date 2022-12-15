import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import numpy as np
from scipy.interpolate import griddata 
from scipy.interpolate import RegularGridInterpolator
import common_funcs as cf
import plotly.express as px

os.chdir('data')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_arsia_no_AC.csv')
# Create powerset of all volcano combinations
p_set = list(cf.powerset(df_volc['Volcano Name']))
p_set.pop(0) #Removes first element which is empty 

# Read in GRS data
df_GRS = pd.read_csv('GRS_data_raw_180.csv')

GRS_lats,GRS_lons,GRS_vals_flattened,GRS_grid = cf.GRS_wrangle(df_GRS)

GRS_masked = np.ma.masked_invalid(GRS_grid)

volc_1_dict = {}
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    # volc_1_dict[volc_name] = xr.open_dataset('indv_volcs\\'+'diagfi_volc_filt_nccomp_'+volc_name+'.nc',mask_and_scale=False,decode_times=False)
    volc_1_dict[volc_name] = np.load(volc_name + '_volc_1_surf'  + '.npy')
        
        

# log plot
fig,axs = plt.subplots(5,4,sharex=True,sharey=True)
axs=axs.flatten()
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    im=axs[cnt].contourf(GRS_lons,GRS_lats,np.ma.masked_array(volc_1_dict[volc_name],GRS_masked.mask),levels = [0.00001e13,.00005e13,.0001e13,.0005e13,.001e13,.005e13,.01e13,.05e13,.1e13,.5e13,1e13],norm=colors.LogNorm())
    axs[cnt].set_title(volc_name)
    axs[cnt].grid(True)
fig.colorbar(im,ax=axs.ravel().tolist())

axs[-1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs[-1].grid(True)

# GRS plot
fig_GRS,ax_GRS = plt.subplots()
im=ax_GRS.pcolormesh(GRS_lons,GRS_lats,GRS_grid)
ax_GRS.set_title("GRS")
ax_GRS.grid(True)



# Non log plot
fig1,axs1 = plt.subplots(5,4,sharex=True,sharey=True)
axs1=axs1.flatten()
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    im=axs1[cnt].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(volc_1_dict[volc_name],GRS_masked.mask))
    axs1[cnt].set_title(volc_name)
    axs1[cnt].grid(True)
    plt.colorbar(im, ax=axs1[cnt])
# fig1.colorbar(im,ax=axs1.ravel().tolist())




all_volc_sum = np.zeros([36,72])
for volc_name in p_set[-1]:
    temp = volc_1_dict[volc_name]
    all_volc_sum += temp

fig2,ax2 = plt.subplots()
im = ax2.pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(volc_1_dict[volc_name],GRS_masked.mask),norm=colors.LogNorm())
ax2.grid(True)
plt.colorbar(im,ax=ax2)
plt.show()


volc1_log = np.log10(volc_1_dict["Hadriaca_Patera"])
log_min = float(volc1_log.min())
log_max = float(volc1_log.max())
plfig = px.imshow(volc_1_dict["Hadriaca_Patera"], animation_frame='Time')
plfig.show()

# Read in one LMD GCM output file ot grab lat/lon geo and create a meshgrid
ds_geo = xr.open_dataset('Indv_volcs\\'+'diagfi_volc_filt_nccomp_Alba_Patera.nc',mask_and_scale=False,decode_times=False)
lat = ds_geo.latitude
lon = ds_geo.longitude
LMD_lon,LMD_lat = np.meshgrid(lon,lat)


# Initialize summed volcanic ash array. 
volc1_sum = np.zeros([97,129])
volc2_sum = np.zeros([97,129])
volc3_sum = np.zeros([97,129])
volc4_sum = np.zeros([97,129])

# Loop through LMD outputs, sum up ash arrays
for volc_name in df_volc['Volcano Name']:
    ds = xr.open_dataset('Indv_volcs\\'+'diagfi_volc_filt_nccomp_'+volc_name+'.nc',mask_and_scale=False,decode_times=False)
    volc1 = ds.volc_1_surf[-1,:,:]
    volc1 = volc1/np.max(volc1)
    volc1_sum += volc1
    
    volc2 = ds.volc_2_surf[-1,:,:]
    volc2 = volc2/np.max(volc2)
    volc2_sum += volc2
    
    volc3 = ds.volc_3_surf[-1,:,:]
    volc3 = volc3/np.max(volc3)
    volc3_sum += volc3
    
    volc4 = ds.volc_4_surf[-1,:,:]
    volc4 = volc4/np.max(volc4)
    volc4_sum += volc4


# Interpolate volc sum array to GRS grid. Plot to check everything is working. 
interp = RegularGridInterpolator((lat,lon),volc1_sum,method="nearest")
volc_interp = interp((GRS_lats,GRS_lons))
plt.pcolormesh(GRS_lons,GRS_lats,volc_interp)



# plt.pcolormesh(lon,lat, volc_sum)


print("pause")

volc_merged = xr.merge([volc1_sum,volc2_sum,volc3_sum,volc4_sum])

volc_merged.to_netcdf('all_volc_dep_norm.nc')

# volc1_sum.to_netcdf('all_volc1_dep.nc')
# volc2_sum.to_netcdf('all_volc2_dep.nc')
# volc3_sum.to_netcdf('all_volc3_dep.nc')
# volc4_sum.to_netcdf('all_volc4_dep.nc')