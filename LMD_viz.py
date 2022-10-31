import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata 
from scipy.interpolate import RegularGridInterpolator


os.chdir('D:\\')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_Arsia.csv')

# Read in GRS data
df_GRS = pd.read_csv('C:\\Users\\palatyle\\Documents\\ArcGIS\\Projects\\Mars_GIS\\data\\GRS_data_raw.csv')

# Sort data and drop Center Lon column
df_GRS_sorted = df_GRS.sort_values(by=['Lon_fix','CenterLat'],ascending=False)
df_GRS_sorted = df_GRS_sorted.drop(columns=['CenterLon'])

# Convert to numpy array and only grab unique vals
GRS_lons = df_GRS_sorted['Lon_fix'].to_numpy()
GRS_lons = np.unique(GRS_lons)

GRS_lats = df_GRS_sorted['CenterLat'].to_numpy()
GRS_lats = np.unique(GRS_lats)
GRS_lats = np.sort(GRS_lats)
GRS_lats = GRS_lats[::-1]

# Create GRS meshgrid
GRS_lon,GRS_lat = np.meshgrid(GRS_lons,GRS_lats)

# Interpolate (really just reshape) GRS data to meshgrid
GRS_vals_grd = griddata((df_GRS['Lon_fix'].to_numpy(),df_GRS['CenterLat'].to_numpy()),df_GRS['Concentration'].to_numpy(),(GRS_lon,GRS_lat),method='nearest')
plt.pcolormesh(GRS_lon,GRS_lat,GRS_vals_grd)
plt.show

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
volc_interp = interp((GRS_lat,GRS_lon))
plt.pcolormesh(GRS_lon,GRS_lat,volc_interp)



# plt.pcolormesh(lon,lat, volc_sum)


print("pause")

volc_merged = xr.merge([volc1_sum,volc2_sum,volc3_sum,volc4_sum])

volc_merged.to_netcdf('all_volc_dep_norm.nc')

# volc1_sum.to_netcdf('all_volc1_dep.nc')
# volc2_sum.to_netcdf('all_volc2_dep.nc')
# volc3_sum.to_netcdf('all_volc3_dep.nc')
# volc4_sum.to_netcdf('all_volc4_dep.nc')