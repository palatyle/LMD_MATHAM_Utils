import os

import common_funcs as cf
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from cmcrameri import cm
from scipy.stats import boxcox

# Change directory to data directory
os.chdir('LMD_MATHAM_Utils/data/no_tharsis')


# Read in GRS data
df_GRS_H2O = pd.read_csv('../GRS_data_raw_180.csv')
df_GRS_S = pd.read_csv('../GRS_data_S_180.csv')

# Wrangle GRS data into list of lats + lons + full array flattened (for use in correlation calc)
GRS_lats,GRS_lons,GRS_H2O_flattened,GRS_grid = cf.GRS_wrangle(df_GRS_H2O)
GRS_H2O_flattened_nn = GRS_H2O_flattened[~np.isnan(GRS_H2O_flattened)]
GRS_masked = np.ma.masked_invalid(GRS_grid)

_,_,GRS_S_flattened,GRS_S_grid = cf.GRS_wrangle(df_GRS_S)
GRS_S_flattened_nn = GRS_S_flattened[~np.isnan(GRS_S_flattened)]


# Log transformation to become normal(ish)
GRS_H2O_flattened_trans_box = boxcox(GRS_H2O_flattened_nn)
GRS_H2O_flattened_trans = GRS_H2O_flattened_trans_box[0]

GRS_S_flattened_trans_box = boxcox(GRS_S_flattened_nn)
GRS_S_flattened_trans = GRS_S_flattened_trans_box[0]


# Read in DCI data
DCI_raw = np.load('../DCI_interp.npy')
DCI_raw_norm = (-DCI_raw-np.nanmin(-DCI_raw))/(np.nanmax(-DCI_raw)-np.nanmin(-DCI_raw))
DCI_flat = DCI_raw.flatten()
neg_DCI_flat = -DCI_flat
DCI_norm = (neg_DCI_flat-np.nanmin(neg_DCI_flat))/(np.nanmax(neg_DCI_flat)-np.nanmin(neg_DCI_flat))

DCI_nn = DCI_flat[~np.isnan(GRS_H2O_flattened)]

DCI_box = boxcox(DCI_flat)
DCI_trans = DCI_box[0]
# DCI_trans = DCI_norm

# Best volc list
best_volcs = ['Apollinaris_Patera', 'Elysium_Mons', 'Cerberus', 'Olympus_Mons', 'Arsia_Mons', 'Pavonis_Mons', 'Ascraeus_Mons', 'Syrtis_Major', 'Hadriaca_Patera', 'Pityusa_Patera', 'Alba_Patera', 'Electris', 'Eden_Patera', 'Siloe_Patera', 'Ismenia_Oxus']



volc_dict={}
volc_dict_flat={}
gdf_dict = {}
gdf_df_dict = {}
for cnt,volc_name in enumerate(best_volcs):
    volc_dict[volc_name] = np.load(volc_name +  "_volc_1_surf.npy")
    
    temp_xarr = cf.arr_to_xarr(volc_dict[volc_name],GRS_lats,GRS_lons,volc_name)
    gdf = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
    gdf.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/' + volc_name + '.gpkg',driver="GPKG")

    gdf_df_dict[volc_name] = pd.DataFrame(gdf)
    
    flat = volc_dict[volc_name].flatten()
    flat_box = boxcox(flat)
    flat_trans = flat_box[0]
    volc_dict_flat[volc_name] = flat_trans
    


volc_dict_flat_df = pd.DataFrame.from_dict(volc_dict_flat)
GRS_H2O_df = pd.DataFrame(GRS_H2O_flattened,columns=['GRS_H2O'])
GRS_S_df = pd.DataFrame(GRS_S_flattened,columns=['GRS_S'])
DCI_df = pd.DataFrame(DCI_trans,columns=['DCI'])

final_volc_regr_df = pd.concat([volc_dict_flat_df,DCI_df,GRS_S_df,GRS_H2O_df],axis=1)
final_volc_regr_df.dropna(how='any',inplace=True)



# fig,ax = plt.subplots()
# plt.scatter(final_volc_regr_df['DCI'],final_volc_regr_df['GRS_H2O'])
# plt.show()

AIC = []
r = []
regressors = []
summary_stat = []

regressor_OLS = sm.OLS(final_volc_regr_df.GRS_H2O,sm.add_constant(final_volc_regr_df.drop(['GRS_H2O'],axis=1))).fit()
# lm = pg.linear_regression(final_volc_regr_df.drop(['GRS_H2O'],axis=1),final_volc_regr_df.GRS_H2O,add_intercept=True,relimp=True)

lm_S = pg.linear_regression(final_volc_regr_df.GRS_S,final_volc_regr_df.GRS_H2O,add_intercept=True,relimp=False)
lm_DCI = pg.linear_regression(final_volc_regr_df.DCI,final_volc_regr_df.GRS_H2O,add_intercept=True,relimp=False)
lm_all = pg.linear_regression(final_volc_regr_df.drop(['GRS_H2O'],axis=1),final_volc_regr_df.GRS_H2O,add_intercept=True,relimp=False)


print(lm_S)
print(lm_DCI)
print(lm_all)


# fig1,ax1 = plt.subplots()
# lm_all.drop(0).sort_values('relimp_perc',ascending=False).plot(x='names',y='relimp_perc',kind='bar',rot=45,ax=ax1)
# ax1.grid(False)
# ax1.set_title('Relative Importance')
# ax1.set_ylabel('% of response variance')
# fig1.savefig('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/figs/OLS_relative_importance_perc_DCI_S.pdf', format = 'pdf',bbox_inches=None, dpi=300)


params = regressor_OLS.params
regression_sum = np.zeros([36,72])
for element,volc_name in enumerate(best_volcs):
    temp = volc_dict[volc_name] * params[element+1]
    regression_sum += temp

regression_sum += DCI_raw * params[16]
regression_sum += GRS_S_grid * params[17]
regression_sum += params[0]

# regression_sum[regression_sum<=0] = 1
fig1,axs1 = plt.subplots(2,1,sharex=True,sharey=True)
axs1=axs1.flatten()

# im=axs1[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(regression_sum,GRS_masked.mask),norm=colors.LogNorm(),edgecolors='none')
im=axs1[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(regression_sum,GRS_masked.mask),norm=colors.SymLogNorm(linthresh=1e10, vmin=np.nanmin(regression_sum),vmax=np.nanmax(regression_sum)),edgecolors='none',cmap=cm.fes)
axs1[0].set_title("OLS everything ")
axs1[0].grid(True)


fig1.colorbar(im,ax=axs1[0],orientation='vertical')

im=axs1[1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs1[1].set_title("GRS")
axs1[1].grid(True)
fig1.colorbar(im,ax=axs1[1],orientation='vertical')


# fig1.savefig(out_dir+"best_volcs_OLS_test.pdf", format = 'pdf',bbox_inches=None, dpi=300)

temp_xarr = cf.arr_to_xarr(np.log10(np.ma.masked_array(regression_sum,GRS_masked.mask)),GRS_lats,GRS_lons,"volc_1_surf")
# temp_xarr = cf.arr_to_xarr(regression_sum,GRS_lats,GRS_lons,"volc_1_surf")

gdf = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/best_reg_volcs_test.gpkg',driver="GPKG")


temp_xarr = cf.arr_to_xarr(GRS_grid,GRS_lats,GRS_lons,"H2O")
gdf_H = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf_H.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/GRS_H2O.gpkg',driver="GPKG")


temp_xarr = cf.arr_to_xarr(GRS_S_grid,GRS_lats,GRS_lons,"S")
gdf_S = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf_S.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/GRS_S.gpkg',driver="GPKG")


temp_xarr = cf.arr_to_xarr(DCI_raw_norm,GRS_lats,GRS_lons,"DCI")
gdf_DCI = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf_DCI.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/DCI.gpkg',driver="GPKG")
print('stop')


temp = pd.concat(gdf_df_dict, axis=1)
temp.to_csv('volc_predictors.csv')

pd.DataFrame(gdf_S).to_csv('S.csv')
pd.DataFrame(gdf_DCI).to_csv('DCI.csv')
pd.DataFrame(gdf_H).to_csv('H.csv')