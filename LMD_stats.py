import pandas as pd
import os
import numpy as np
import time
import common_funcs as cf

# Start timer
t0 = time.time()

# Change directory to data directory
os.chdir('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_Arsia.csv')

# Create powerset of all volcano combinations
p_set = list(cf.powerset(df_volc['Volcano Name']))
p_set.pop(0) #Removes first element which is empty 

# Read in GRS data
df_GRS = pd.read_csv('GRS_data_raw_180.csv')

# Wrangle GRS data into list of lats + lons + full array flattened (for use in correlation calc)
GRS_lats,GRS_lons,GRS_vals_flattened = cf.GRS_wrangle(df_GRS)

# Define tracer names to loop through
tracer_names = ['volc_1_surf','volc_2_surf','volc_3_surf','volc_4_surf',]

# Define empty dictionaries for each tracer
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


# Initialize empty lists for r and Shapiro-Wilks values
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
                temp = volc_1_dict[name]
                volc_sum_1 += temp
            elif tracer == 'volc_2_surf':
                temp = volc_2_dict[name]
                volc_sum_2 += temp
            elif tracer == 'volc_3_surf':
                temp = volc_3_dict[name]
                volc_sum_3 += temp
            elif tracer == 'volc_4_surf':
                temp = volc_4_dict[name]
                volc_sum_4 += temp
        # With sum finished, calcualte r and shapiro wilkes test for every summed array
        if tracer == 'volc_1_surf':
            temp_r, temp_sp = cf.get_r_val(volc_sum_1,GRS_vals_flattened)
            r_volc_1.append(temp_r)
            sp_volc_1.append(temp_sp)
        elif tracer == 'volc_2_surf':
            temp_r, temp_sp = cf.get_r_val(volc_sum_2,GRS_vals_flattened)
            r_volc_2.append(temp_r)
            sp_volc_2.append(temp_sp)
        elif tracer == 'volc_3_surf':
            temp_r, temp_sp = cf.get_r_val(volc_sum_3,GRS_vals_flattened)
            r_volc_3.append(temp_r)
            sp_volc_3.append(temp_sp)
        elif tracer == 'volc_4_surf':
            temp_r, temp_sp = cf.get_r_val(volc_sum_4,GRS_vals_flattened)
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

best_set = p_set[df_volc_1.r_val.idxmax()]
best_set_df = pd.DataFrame(best_set)
best_set_df.columns = ['Volcano Name']
pd.merge(best_set_df,df_volc[["Volcano Name","lat","lon"]],on="Volcano Name",how="left")
best_set_df.to_csv('best_volcs.csv')

# df_volc_1.r_val.idxmax()
print(best_set) 
print('r: '+ str(df_volc_1['r_val'][df_volc_1.r_val.idxmax()]))
print('r P val: ' + str(df_volc_1['r-p_val'][df_volc_1.r_val.idxmax()]))
print('SP: ' + str(df_volc_1['sp_val'][df_volc_1.r_val.idxmax()]))
print('SP p val: ' + str(df_volc_1['sp-p_val'][df_volc_1.r_val.idxmax()]))

print("All Volcs:") 
print('r: '+ str(df_volc_1['r_val'].iat[-1]))
print('r P val: ' + str(df_volc_1['r-p_val'].iat[-1]))
print('SP: ' + str(df_volc_1['sp_val'].iat[-1]))
print('SP p val: ' + str(df_volc_1['sp-p_val'].iat[-1]))


# For only best volcs
best_volc_sum = np.zeros([36,72])
for volc_name in best_set:
    temp = volc_1_dict[volc_name]
    best_volc_sum += temp

temp_xarr = cf.arr_to_xarr(best_volc_sum,GRS_lats,GRS_lons,"volc_1_surf_norm")
gdf = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf.to_file('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data\\best_volcs.gpkg',driver="GPKG")


# For all volcs
all_volc_sum = np.zeros([36,72])
for volc_name in p_set[-1]:
    temp = volc_1_dict[volc_name]
    all_volc_sum += temp

temp_xarr = cf.arr_to_xarr(all_volc_sum,GRS_lats,GRS_lons,"volc_1_surf_norm")
gdf = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf.to_file('C:\\Users\\palatyle\\Documents\\LMD_MATHAM_Utils\\data\\all_volcs.gpkg',driver="GPKG")


print('Done in '+ str(t1-t0) + ' seconds')
print('stop')