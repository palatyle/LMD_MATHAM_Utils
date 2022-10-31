import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata 
from scipy.interpolate import RegularGridInterpolator
import scipy.stats
from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

os.chdir('D:\\')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_Arsia.csv')

p_set = list(powerset(df_volc['Volcano Name']))

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
ds_volc = xr.open_dataset('all_volc_dep_norm.nc',mask_and_scale=False,decode_times=False)
lat = ds_volc.latitude
lon = ds_volc.longitude
LMD_lon,LMD_lat = np.meshgrid(lon,lat)

# Interpolate volc sum array to GRS grid. Plot to check everything is working. 
interp = RegularGridInterpolator((lat,lon),ds_volc.volc_1_surf,method="nearest")
volc_interp = interp((GRS_lat,GRS_lon))
plt.pcolormesh(GRS_lon,GRS_lat,volc_interp)
# plt.show()

GRS_flat = GRS_vals_grd.reshape(GRS_vals_grd.shape[0]*GRS_vals_grd.shape[1])
volc_interp_flat = volc_interp.reshape(volc_interp.shape[0]*volc_interp.shape[1])

volc_interp_flat_nn = volc_interp_flat[~np.isnan(GRS_flat)]
GRS_flat_nn = GRS_flat[~np.isnan(GRS_flat)]

r = scipy.stats.pearsonr(GRS_flat_nn, volc_interp_flat_nn)
print(r)
print('stop')