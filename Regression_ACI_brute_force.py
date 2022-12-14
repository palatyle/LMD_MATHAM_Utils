import pandas as pd
import os
import numpy as np
import time
import common_funcs as cf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, boxcox, probplot, pearsonr, spearmanr, normaltest


# Start timer
t0 = time.time()

# Change directory to data directory
os.chdir('data')

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs_no_Arsia.csv')

# Create powerset of all volcano combinations
p_set = list(cf.powerset(df_volc['Volcano Name']))
p_set.pop(0) #Removes first element which is empty 

# Read in GRS data
df_GRS = pd.read_csv('GRS_data_raw_180.csv')

# Wrangle GRS data into list of lats + lons + full array flattened (for use in correlation calc)
GRS_lats,GRS_lons,GRS_vals_flattened,GRS_grid = cf.GRS_wrangle(df_GRS)
GRS_vals_flattened_nn = GRS_vals_flattened[~np.isnan(GRS_vals_flattened)]

# Log transformation to become normal(ish)
GRS_vals_flattened_trans = boxcox(GRS_vals_flattened_nn,0)

tracer_names = ['volc_1_surf']

# Define empty dictionaries for each tracer
volc_1_dict = {}
volc_1_dict_flat = {}
volc_2_dict = {}
volc_3_dict = {}
volc_4_dict = {}


prev_max = 0
prev_min = 1e16
# Loop through all separate GCM outputs and read in. For each volcanic tracer, interpolate from the GCM grid to the GRS grid
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    for tracer in tracer_names: 
        if tracer == 'volc_1_surf':
            volc_1_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
            max_val = np.max(volc_1_dict[volc_name])
            
            if max_val > prev_max:
                prev_max = max_val
            min_val = np.min(volc_1_dict[volc_name])
            if min_val < prev_min:
                prev_min = min_val
            temp =  volc_1_dict[volc_name].flatten()
            temp2 = boxcox(temp)
            temp3 = temp2[0]
            print(temp2[1])
            volc_1_dict_flat[volc_name] = temp3[~np.isnan(GRS_vals_flattened)]
        elif tracer == 'volc_2_surf':
            volc_2_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        elif tracer == 'volc_3_surf':
            volc_3_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        elif tracer == 'volc_4_surf':
            volc_4_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        print("done with tracer: "+tracer+" and volcano: " + volc_name)
fig,axs = plt.subplots(5,4,sharex=True,sharey=True)
axs=axs.flatten()
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    im=axs[cnt].pcolormesh(GRS_lons,GRS_lats,volc_1_dict[volc_name],norm=colors.LogNorm())
    axs[cnt].set_title(volc_name)
    axs[cnt].grid(True)
fig.colorbar(im,ax=axs.ravel().tolist())
all_volc_sum = np.zeros([36,72])
for volc_name in p_set[-1]:
    temp = volc_1_dict[volc_name]
    all_volc_sum += temp

im=axs[-1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs[-1].set_title("GRS")
axs[-1].grid(True)




fig1,axs1 = plt.subplots(5,4,sharex=True,sharey=True)
axs1=axs1.flatten()
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    im=axs1[cnt].pcolormesh(GRS_lons,GRS_lats,volc_1_dict[volc_name])
    axs1[cnt].set_title(volc_name)
    axs1[cnt].grid(True)
    plt.colorbar(im, ax=axs1[cnt])
# fig1.colorbar(im,ax=axs1.ravel().tolist())
all_volc_sum = np.zeros([36,72])
for volc_name in p_set[-1]:
    temp = volc_1_dict[volc_name]
    all_volc_sum += temp

im=axs1[-1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs1[-1].set_title("GRS")
axs1[-1].grid(True)

plt.colorbar(im,ax=axs1[-1])


fig2,ax2 = plt.subplots()
im = ax2.pcolormesh(GRS_lons,GRS_lats,all_volc_sum,norm=colors.LogNorm())
ax2.grid(True)
plt.colorbar(im,ax=ax2)
plt.show()
volc_1_dict_flat_df = pd.DataFrame.from_dict(volc_1_dict_flat)

AIC = []
summary_stat = []

for count,set_loop in enumerate(p_set):
    # regressor_OLS = sm.OLS(GRS_vals_flattened_trans,volc_1_dict_flat_df[list(set_loop)].assign(intercept=1)).fit()
    regressor_OLS = sm.OLS(GRS_vals_flattened_trans,sm.add_constant(volc_1_dict_flat_df[list(set_loop)])).fit()
    AIC.append(regressor_OLS.aic)
    summary_stat.append(regressor_OLS.summary())
    if count % 10000 == 0:
        print(" set: " + str(count) + '/' + str(len(p_set)))

print(p_set[np.argmin(AIC)])
print('done')