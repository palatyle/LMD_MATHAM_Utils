import os

import common_funcs as cf
import numpy as np
import pandas as pd
import xarray as xr

os.chdir('/Volumes/MATHAM4/no_tharsis')

# Read in volcano location data
df_volc = pd.read_csv('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/Mars_Volc_locs.csv')

# Read in GRS data
df_GRS = pd.read_csv('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/GRS_data_raw_180.csv')

# Wrangle GRS data into list of lats + lons + full array flattened (for use in correlation calc)
GRS_lats,GRS_lons,GRS_vals_flattened,GRS_grid = cf.GRS_wrangle(df_GRS)

# Create GRS meshgrid
GRS_lon,GRS_lat = np.meshgrid(GRS_lons,GRS_lats)

# Read in one LMD GCM output file ot grab lat/lon geo and create a meshgrid
ds_volc = xr.open_dataset('Alba_Patera_diagfi_volc_filt.nccomp.nc',mask_and_scale=False,decode_times=False)
lat = ds_volc.latitude
lon = ds_volc.longitude
LMD_lon,LMD_lat = np.meshgrid(lon,lat)

  
tracer_names = ['volc_1_surf','volc_2_surf','volc_3_surf','volc_4_surf',]

# Loop through all separate GCM outputs and read in. For each volcanic tracer, interpolate from the GCM grid to the GRS grid
for volc_name in df_volc['Volcano Name']:
    # Use xarray to open GCM output for each volcano
    temp = xr.open_dataset(volc_name+'_diagfi_volc_filt.nccomp.nc',mask_and_scale=False,decode_times=False)
    # Loop through tracers
    for tracer in tracer_names: 
        # Interpolate from GCM grid to coarser GRS grid
        volc_1= cf.interpolate_data(lat,lon,GRS_lat,GRS_lon,temp[tracer][-1,:,:])
        
        # Save array as a numpy file
        np.save('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/no_tharsis/' + volc_name + '_' + tracer, volc_1, allow_pickle=False)

        # Convert array to xarray object
        if tracer == 'volc_1_surf':
            volc1_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
        if tracer == 'volc_2_surf':
            volc2_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
        if tracer == 'volc_3_surf':
            volc3_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
        if tracer == 'volc_4_surf':
            volc4_xr = cf.arr_to_xarr(volc_1,GRS_lats,GRS_lons,tracer)
            
        print("done with tracer: "+tracer+" and volcano: " + volc_name)
    
    # Merge all xarray objects into one, convert to geodataframe, and output as a geopackage
    volc_merged = xr.merge([volc1_xr,volc2_xr,volc3_xr,volc4_xr])
    gdf = cf.xr_to_geodf(volc_merged,"ESRI:104971") # Mars 2000 Sphere coord system
    gdf.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/no_tharsis/' + volc_name +'_interp.gpkg',driver="GPKG")