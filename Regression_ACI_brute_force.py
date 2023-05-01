import os
import time

import common_funcs as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy.stats import boxcox

# Start timer
t0 = time.time()

# Change directory to data directory
os.chdir('LMD_MATHAM_Utils/data/no_tharsis')

# Read in volcano location data
df_volc = pd.read_csv('../Mars_Volc_locs_no_AC.csv')

# Create powerset of all volcano combinations
p_set = list(cf.powerset(df_volc['Volcano Name']))
p_set.pop(0) #Removes first element which is empty 

# Read in GRS data
df_GRS = pd.read_csv('../GRS_data_raw_180.csv')

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
# Loop through all separate GCM outputs and read in.
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
            # volc_1_dict_flat[volc_name] = temp[~np.isnan(GRS_vals_flattened)]

        elif tracer == 'volc_2_surf':
            volc_2_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        elif tracer == 'volc_3_surf':
            volc_3_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        elif tracer == 'volc_4_surf':
            volc_4_dict[volc_name] = np.load(volc_name + '_' + tracer + '.npy')
        print("done with tracer: "+tracer+" and volcano: " + volc_name)


volc_1_dict_flat_df = pd.DataFrame.from_dict(volc_1_dict_flat)
GRS_flat_df = pd.DataFrame(GRS_vals_flattened[~np.isnan(GRS_vals_flattened)],columns=['GRS'])

final_volc_1_df = pd.concat([volc_1_dict_flat_df,GRS_flat_df],axis=1)

AIC = []
r = []
regressors = []
summary_stat = []

for count,set_loop in enumerate(p_set):
    # regressor_OLS = sm.OLS(GRS_vals_flattened_trans,volc_1_dict_flat_df[list(set_loop)].assign(intercept=1)).fit()
    # regressor_OLS = sm.OLS(zscore(GRS_vals_flattened_trans),sm.add_constant(zscore(volc_1_dict_flat_df[list(set_loop)]))).fit()
    regressor_OLS = sm.OLS(GRS_vals_flattened_trans,sm.add_constant(volc_1_dict_flat_df[list(set_loop)])).fit()
    # lm = pg.linear_regression(volc_1_dict_flat_df[list(set_loop)],GRS_vals_flattened_trans,add_intercept=True,relimp=True)
    # summary_stat.append(lm)
    # regressors.append(regressor_OLS)
    results_summary = regressor_OLS.summary()

    results_as_html = results_summary.tables[1].as_html()
    summary_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    AIC.append(regressor_OLS.aic)
    r.append(regressor_OLS.rsquared)
    summary_stat.append(summary_df)
    if count % 10000 == 0:
        print(" set: " + str(count) + '/' + str(len(p_set)))

print(p_set[np.argmin(AIC)])
r[np.argmin(AIC)]
np.min(AIC)

lm = pg.linear_regression(zscore(volc_1_dict_flat_df[list(p_set[np.argmin(AIC)])]),zscore(GRS_vals_flattened_trans),add_intercept=True,relimp=True)

lm_all = pg.linear_regression(zscore(volc_1_dict_flat_df[list(p_set[-1])]),zscore(GRS_vals_flattened_trans),add_intercept=True,relimp=True)

print('done')

fig1,ax1 = plt.subplots()
lm.drop(0).sort_values('relimp_perc',ascending=False).plot(x='names',y='relimp_perc',kind='bar',rot=45,ax=ax1)
ax1.grid(False)
ax1.set_title('Relative Importance')
ax1.set_ylabel('% of response variance')
fig1.savefig('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/figs/OLS_relative_importance_perc.pdf', format = 'pdf',bbox_inches=None, dpi=300)


fig2,ax2 = plt.subplots()
summary_stat[np.argmin(AIC)]['Names'] = summary_stat[np.argmin(AIC)].index
summary_stat[np.argmin(AIC)]['coef'] = summary_stat[np.argmin(AIC)]['coef'].abs()
summary_stat[np.argmin(AIC)].drop(['const']).sort_values('coef',ascending=False).plot(x='Names',y='coef',kind='bar',rot=45,ax=ax2)
ax2.grid(False)
ax2.set_title('Relative Importance')
ax2.set_ylabel('Coeff value')
fig2.savefig('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/figs/OLS_relative_importance_beta_coef.pdf', format = 'pdf',bbox_inches=None, dpi=300)


fig3,ax3 = plt.subplots()
lm_all.drop(0).sort_values('relimp_perc',ascending=False).plot(x='names',y='relimp_perc',kind='bar',rot=45,ax=ax3)
ax3.grid(False)
ax3.set_title('Relative Importance')
ax3.set_ylabel('% of response variance')
fig3.savefig('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/figs/OLS_relative_importance_perc.pdf', format = 'pdf',bbox_inches=None, dpi=300)
