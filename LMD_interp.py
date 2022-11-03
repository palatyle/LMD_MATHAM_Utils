import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata 
from scipy.interpolate import RegularGridInterpolator
import scipy.stats
from itertools import chain, combinations
import time

def interpolate_data(X,Y,Xi,Yi,arr):
    # Set up interpolator object which has original grid data
    interpolator = RegularGridInterpolator((X,Y),arr,method="nearest")
    # Interpolate to Xi,Yi grid
    return interpolator((Xi,Yi))

os.chdir('D:\\')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_Arsia.csv')

# Read in GRS data
df_GRS = pd.read_csv('C:\\Users\\palatyle\\Documents\\ArcGIS\\Projects\\Mars_GIS\\data\\GRS_data_raw_180.csv')

# Sort data and drop Center Lon column
df_GRS_sorted = df_GRS.sort_values(by=['lon_180','CenterLat'],ascending=False)
df_GRS_sorted = df_GRS_sorted.drop(columns=['CenterLon'])

# Convert to numpy array and only grab unique vals
GRS_lons = df_GRS_sorted['lon_180'].to_numpy()
GRS_lons = np.unique(GRS_lons)

GRS_lats = df_GRS_sorted['CenterLat'].to_numpy()
GRS_lats = np.unique(GRS_lats)
GRS_lats = np.sort(GRS_lats)
GRS_lats = GRS_lats[::-1]

# Create GRS meshgrid
GRS_lon,GRS_lat = np.meshgrid(GRS_lons,GRS_lats)

# Read in one LMD GCM output file ot grab lat/lon geo and create a meshgrid
ds_volc = xr.open_dataset('all_volc_dep_norm.nc',mask_and_scale=False,decode_times=False)
lat = ds_volc.latitude
lon = ds_volc.longitude
LMD_lon,LMD_lat = np.meshgrid(lon,lat)

os.chdir('Indv_volcs')    
tracer_names = ['volc_1_surf','volc_2_surf','volc_3_surf','volc_4_surf',]
# volc_interp = interpolate_data(lat,lon,GRS_lat,GRS_lon,ds_volc.volc_1_surf)
# r = get_r_val(volc_interp,GRS_vals_grd)



# Loop through all separate GCM outputs and read in. For each volcanic tracer, interpolate from the GCM grid to the GRS grid
for volc_name in df_volc['Volcano Name']:
    temp = xr.open_dataset('diagfi_volc_filt_nccomp_' + volc_name + '.nc',mask_and_scale=False,decode_times=False)
    for tracer in tracer_names: 
        volc_1= interpolate_data(lat,lon,GRS_lat,GRS_lon,temp[tracer][-1,:,:])
        np.save('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data\\' + volc_name + '_' + tracer, volc_1, allow_pickle=False)
        print("done with tracer: "+tracer+" and volcano: " + volc_name)
        if tracer == 'volc_1_surf':
            volc1_xr = xr.DataArray(data = volc_1,coords={'latitude': GRS_lats,'longitude': GRS_lons},dims=['latitude','longitude'],name=tracer)
        if tracer == 'volc_2_surf':
            volc2_xr = xr.DataArray(data = volc_1,coords={'latitude': GRS_lats,'longitude': GRS_lons},dims=['latitude','longitude'],name=tracer)
        if tracer == 'volc_3_surf':
            volc3_xr = xr.DataArray(data = volc_1,coords={'latitude': GRS_lats,'longitude': GRS_lons},dims=['latitude','longitude'],name=tracer)    
        if tracer == 'volc_4_surf':
            volc4_xr = xr.DataArray(data = volc_1,coords={'latitude': GRS_lats,'longitude': GRS_lons},dims=['latitude','longitude'],name=tracer)
            
    volc_merged = xr.merge([volc1_xr,volc2_xr,volc3_xr,volc4_xr])
    volc_merged_df = volc_merged.to_dataframe()
    volc_merged_df.to_csv('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data\\' + volc_name +'_interp.csv')
    # volc_merged.to_netcdf('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data\\' + volc_name +'_interp.nc')