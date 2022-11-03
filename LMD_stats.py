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


t0 = time.time()
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

    
os.chdir('data')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_Arsia.csv')

p_set = list(powerset(df_volc['Volcano Name']))
p_set.pop(0) #Removes first element which is empty 

# Read in GRS data
df_GRS = pd.read_csv('GRS_data_raw_180.csv')

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

# Interpolate (really just reshape) GRS data to meshgrid
GRS_vals_grd = griddata((df_GRS['lon_180'].to_numpy(),df_GRS['CenterLat'].to_numpy()),df_GRS['Concentration'].to_numpy(),(GRS_lon,GRS_lat),method='nearest')
GRS_vals_flattened = GRS_vals_grd.reshape(GRS_vals_grd.shape[0]*GRS_vals_grd.shape[1])

def interpolate_data(X,Y,Xi,Yi,arr):
    # Set up interpolator object which has original grid data
    interpolator = RegularGridInterpolator((X,Y),arr,method="nearest")
    # Interpolate to Xi,Yi grid
    return interpolator((Xi,Yi))

def get_r_val(arr,arr_nans_flat):
    # Reshape to flat array
    arr_flat = arr.reshape(arr.shape[0]*arr.shape[1])
    # Remove nans from both arrays
    arr_flat_nn = arr_flat[~np.isnan(arr_nans_flat)]
    arr_nans_flat_nn = arr_nans_flat[~np.isnan(arr_nans_flat)]
    
    r = scipy.stats.pearsonr(arr_nans_flat_nn,arr_flat_nn)
    sp = scipy.stats.shapiro(arr_flat_nn)
    return r, sp



tracer_names = ['volc_1_surf','volc_2_surf','volc_3_surf','volc_4_surf',]
# volc_interp = interpolate_data(lat,lon,GRS_lat,GRS_lon,ds_volc.volc_1_surf)
# r = get_r_val(volc_interp,GRS_vals_grd)
volc_1_dict = {}
volc_2_dict = {}
volc_3_dict = {}
volc_4_dict = {}

# Loop through all separate GCM outputs and read in. For each volcanic tracer, interpolate from the GCM grid to the GRS grid
for volc_name in df_volc['Volcano Name']:
    for tracer in tracer_names: 
        if tracer == 'volc_1_surf':
            volc_1_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        elif tracer == 'volc_2_surf':
            volc_2_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        elif tracer == 'volc_3_surf':
            volc_3_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        elif tracer == 'volc_4_surf':
            volc_4_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        print("done with tracer: "+tracer+" and volcano: " + volc_name)


# Initialize empty lists
r_volc_1 = []
r_volc_2 = []
r_volc_3 = []
r_volc_4 = []

sp_volc_1 = []
sp_volc_2 = []
sp_volc_3 = []
sp_volc_4 = []

# Loop through the 4 volcanic tracers
for tracer in tracer_names:
    # Loop through the superset
    for count, set in enumerate(p_set):
        # Initialize the sum array with zeros (this should reset for every set of the superset)
        volc_sum_1 = np.zeros([36,72])
        volc_sum_2 = np.zeros([36,72])
        volc_sum_3 = np.zeros([36,72])
        volc_sum_4 = np.zeros([36,72])
        # Loop through each volcano name in each set and normalize the data and sum if more than 1 volcano is present in set. 
        for name in set:
            if tracer == 'volc_1_surf':
                temp = volc_1_dict[name]/np.max(volc_1_dict[name])
                volc_sum_1 += temp
            elif tracer == 'volc_2_surf':
                temp = volc_2_dict[name]/np.max(volc_2_dict[name])
                volc_sum_2 += temp
            elif tracer == 'volc_3_surf':
                temp = volc_3_dict[name]/np.max(volc_3_dict[name])
                volc_sum_3 += temp
            elif tracer == 'volc_4_surf':
                temp = volc_4_dict[name]/np.max(volc_4_dict[name])
                volc_sum_4 += temp
        # With sum finished, calcualte r and shapiro wilkes test for every summed array
        if tracer == 'volc_1_surf':
            temp_r, temp_sp = get_r_val(volc_sum_1,GRS_vals_flattened)
            r_volc_1.append(temp_r)
            sp_volc_1.append(temp_sp)
        elif tracer == 'volc_2_surf':
            temp_r, temp_sp = get_r_val(volc_sum_2,GRS_vals_flattened)
            r_volc_2.append(temp_r)
            sp_volc_2.append(temp_sp)
        elif tracer == 'volc_3_surf':
            temp_r, temp_sp = get_r_val(volc_sum_3,GRS_vals_flattened)
            r_volc_3.append(temp_r)
            sp_volc_3.append(temp_sp)
        elif tracer == 'volc_4_surf':
            temp_r, temp_sp = get_r_val(volc_sum_4,GRS_vals_flattened)
            r_volc_4.append(temp_r)
            sp_volc_4.append(temp_sp)
        print("tracer: " + tracer +" set: " + str(count) + '/' + str(len(p_set)))

t1 = time.time()
# # Interpolate volc sum array to GRS grid. Plot to check everything is working. 
# interp = RegularGridInterpolator((lat,lon),ds_volc.volc_1_surf,method="nearest")
# volc_interp = interp((GRS_lat,GRS_lon))
# plt.pcolormesh(GRS_lon,GRS_lat,volc_interp)
# # plt.show()

# GRS_flat = GRS_vals_grd.reshape(GRS_vals_grd.shape[0]*GRS_vals_grd.shape[1])
# volc_interp_flat = volc_interp.reshape(volc_interp.shape[0]*volc_interp.shape[1])

# volc_interp_flat_nn = volc_interp_flat[~np.isnan(GRS_flat)]
# GRS_flat_nn = GRS_flat[~np.isnan(GRS_flat)]

# r = scipy.stats.pearsonr(GRS_flat_nn, volc_interp_flat_nn)
# print(r)


r_volc_1_df = pd.DataFrame(r_volc_1,columns=['r_val','r-p_val'])
sp_volc_1_df = pd.DataFrame(sp_volc_1,columns=['sp_val','sp-p_val'])
df_volc_1 = pd.merge(r_volc_1_df,sp_volc_1_df,left_index=True,right_index=True)

r_volc_2_df = pd.DataFrame(r_volc_2,columns=['r_val','r-p_val'])
sp_volc_2_df = pd.DataFrame(sp_volc_2,columns=['sp_val','sp-p_val'])
df_volc_2 = pd.merge(r_volc_2_df,sp_volc_2_df,left_index=True,right_index=True)

r_volc_3_df = pd.DataFrame(r_volc_3,columns=['r_val','r-p_val'])
sp_volc_3_df = pd.DataFrame(sp_volc_3,columns=['sp_val','sp-p_val'])
df_volc_3 = pd.merge(r_volc_3_df,sp_volc_3_df,left_index=True,right_index=True)

r_volc_4_df = pd.DataFrame(r_volc_4,columns=['r_val','r-p_val'])
sp_volc_4_df = pd.DataFrame(sp_volc_4,columns=['sp_val','sp-p_val'])
df_volc_4 = pd.merge(r_volc_4_df,sp_volc_4_df,left_index=True,right_index=True)

print(p_set[df_volc_1.r_val.idxmax()])

print('Done in '+ str(t1-t0) + ' seconds')
print('stop')