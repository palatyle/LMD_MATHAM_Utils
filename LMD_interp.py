import xarray as xr
import pandas as pd
import os
import numpy as np
import time
import common_funcs as cf





os.chdir('D:\\')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_Arsia.csv')

# Read in GRS data
df_GRS = pd.read_csv('C:\\Users\\palatyle\\Documents\\ArcGIS\\Projects\\Mars_GIS\\data\\GRS_data_raw_180.csv')

# Wrangle GRS data into list of lats + lons + full array flattened (for use in correlation calc)
GRS_lats,GRS_lons,GRS_vals_flattened = cf.GRS_wrangle(df_GRS)

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
        volc_1= cf.interpolate_data(lat,lon,GRS_lat,GRS_lon,temp[tracer][-1,:,:])
        np.save('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data\\' + volc_name + '_' + tracer, volc_1, allow_pickle=False)
        print("done with tracer: "+tracer+" and volcano: " + volc_name)
        if tracer == 'volc_1_surf':
            volc1_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
        if tracer == 'volc_2_surf':
            volc2_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
        if tracer == 'volc_3_surf':
            volc3_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
        if tracer == 'volc_4_surf':
            volc4_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
            
    volc_merged = xr.merge([volc1_xr,volc2_xr,volc3_xr,volc4_xr])
    gdf = cf.xr_to_geodf(volc_merged,"ESRI:104971") # Mars 2000 Sphere coord system
    gdf.to_file('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data\\' + volc_name +'_interp.gpkg',driver="GPKG")
    