import os

import common_funcs as cf
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from cmcrameri import cm
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import boxcox

plt.rcParams.update({'font.sans-serif':'Myriad Pro'})
plt.rcParams.update({'xtick.labelsize': 16, 
                     'ytick.labelsize': 16,
                     'axes.titlesize': 22,
                     'figure.titlesize': 16,
                     'axes.labelsize': 16,
                     'axes.labelsize': 16,
                     'legend.fontsize': 14,
                     'legend.title_fontsize': 14,
                     'figure.facecolor':(240/255,240/255,240/255),
                     'savefig.facecolor':(240/255,240/255,240/255)})

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.chdir('LMD_MATHAM_Utils/data')
out_dir = '/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/figs/'

# Read in volcano location data
df_volc = pd.read_csv('Mars_Volc_locs.csv')
# Create powerset of all volcano combinations
p_set = list(cf.powerset(df_volc['Volcano Name']))
p_set.pop(0) #Removes first element which is empty 

# Read in GRS data
df_GRS = pd.read_csv('GRS_data_raw_180.csv')

GRS_lats,GRS_lons,GRS_vals_flattened,GRS_grid = cf.GRS_wrangle(df_GRS)
GRS_vals_flattened_nn = GRS_vals_flattened[~np.isnan(GRS_vals_flattened)]
GRS_vals_flattened_trans = boxcox(GRS_vals_flattened_nn,0)

GRS_masked = np.ma.masked_invalid(GRS_grid)

os.chdir('no_tharsis')

volc_1_dict = {}
volc_1_dict_reshaped = {}
volc_1_dict_flat_trans = {}
max_vals = []
min_vals = []
gdf_df_dict = {}
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    # volc_1_dict[volc_name] = xr.open_dataset('indv_volcs\\'+'diagfi_volc_filt_nccomp_'+volc_name+'.nc',mask_and_scale=False,decode_times=False)
    volc_1_dict[volc_name]= np.load(volc_name + '_volc_1_surf'  + '.npy')
    
     
    temp_xarr = cf.arr_to_xarr(volc_1_dict[volc_name],GRS_lats,GRS_lons,volc_name)
    gdf = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
    # gdf.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/' + volc_name + '.gpkg',driver="GPKG")

    gdf_df_dict[volc_name] = pd.DataFrame(gdf)
    
    
    
    max_vals.append(volc_1_dict[volc_name].max())
    min_vals.append(volc_1_dict[volc_name].min())
    temp =  volc_1_dict[volc_name].flatten()
    temp2 = boxcox(temp)
    temp3 = temp2[0]
    volc_1_dict_reshaped[volc_name] = temp.reshape((36,72))
    volc_1_dict_flat_trans[volc_name] = temp3[~np.isnan(GRS_vals_flattened)]

temp = pd.concat(gdf_df_dict, axis=1)
temp.to_csv('volc_predictors_all.csv')

# volc_1_dict_df = pd.DataFrame.from_dict(volc_1_dict_flat)
volc_1_dict_trans_df = pd.DataFrame.from_dict(volc_1_dict_flat_trans)

GRS_flat_df = pd.DataFrame(GRS_vals_flattened[~np.isnan(GRS_vals_flattened)],columns=['GRS'])

# final_volc_1_df = pd.concat([volc_1_dict_df,GRS_flat_df],axis=1)

volc_1_dict_no_AC = volc_1_dict.copy()

del volc_1_dict_no_AC['Malea_Patera'], volc_1_dict_no_AC['Peneus_Patera'], volc_1_dict_no_AC['Amphritites']

# log plot contours
fig,axs = plt.subplots(5,4,sharex=True,sharey=True)
axs=axs.flatten()
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    im=axs[cnt].contourf(GRS_lons,GRS_lats,np.ma.masked_array(volc_1_dict[volc_name],GRS_masked.mask),levels = [0.00001e13,.00005e13,.0001e13,.0005e13,.001e13,.005e13,.01e13,.05e13,.1e13,.5e13,1e13],cmap='tab10',norm=colors.LogNorm())
    axs[cnt].set_title(volc_name)
    axs[cnt].grid(True)
fig.colorbar(im,ax=axs.ravel().tolist())

axs[-1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs[-1].grid(True)
axs[-1].set_title("GRS")
axs[-1].grid(True)

# log plot pcolor
fig2,axs2 = plt.subplots(5,4,sharex=True,sharey=True)
axs2=axs2.flatten()
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    im=axs2[cnt].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(volc_1_dict[volc_name],GRS_masked.mask),norm=colors.LogNorm(),edgecolors='none')
    axs2[cnt].set_title(volc_name)
    axs2[cnt].grid(True)
    im.set_clim(min(min_vals),max(max_vals))

fig2.colorbar(im,ax=axs2.ravel().tolist(),label='Ash Area Density (kg/m^2)')
fig2.suptitle('Deposited Ash for Each Volcano')
fig2.supxlabel('Longitude')
fig2.supylabel('Latitude')

# axs2[-1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
# axs2[-1].grid(True)
# axs2[-1].set_title("GRS")
# axs2[-1].grid(True)
manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
fig2.savefig(out_dir+"all_volcs_log_pcolor.pdf", format = 'pdf',bbox_inches=None, dpi=300)



# Regression results
regression_volcs = ['Apollinaris_Patera', 'Elysium_Mons', 'Cerberus', 'Olympus_Mons', 'Arsia_Mons', 'Pavonis_Mons', 'Ascraeus_Mons', 'Syrtis_Major', 'Hadriaca_Patera', 'Pityusa_Patera', 'Alba_Patera', 'Electris', 'Eden_Patera', 'Siloe_Patera', 'Ismenia_Oxus']
regressor_OLS = sm.OLS(GRS_vals_flattened_trans,sm.add_constant(volc_1_dict_trans_df[regression_volcs])).fit()
params = regressor_OLS.params
# params[params<=0] = 0

regr_locs = df_volc[df_volc["Volcano Name"].isin(regression_volcs)].reset_index()
regression_sum = np.zeros([36,72])
for element,volc_name in enumerate(regression_volcs):
    temp = volc_1_dict[volc_name] * params[element+1]
    regression_sum += temp

regression_sum += params[0]

# regression_sum[regression_sum<=0] = 1
fig3,axs3 = plt.subplots(2,1,sharex=True,sharey=True)
axs3=axs3.flatten()

im=axs3[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(regression_sum,GRS_masked.mask),norm=colors.LogNorm(),edgecolors='none')
# im=axs3[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(regression_sum,GRS_masked.mask),norm=colors.SymLogNorm(linthresh=1e10,linscale = 0.5, vmin=np.min(regression_sum),vmax=np.max(regression_sum)),edgecolors='none',cmap=cm.berlin)
axs3[0].set_title("OLS regression volc sum. r^2 = .306. AIC = -257.7 ")
axs3[0].grid(True)
axs3[0].scatter(regr_locs['lon'],regr_locs['lat'],c='red',marker='^')


# for i,txt in enumerate(regression_volcs):
#     axs3[0].annotate(txt,(regr_locs['lon'][i],regr_locs['lat'][i]))
    
fig3.colorbar(im,ax=axs3[0],orientation='vertical')

im=axs3[1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs3[1].set_title("GRS")
axs3[1].grid(True)
fig3.colorbar(im,ax=axs3[1],orientation='vertical')


fig3.savefig(out_dir+"best_volcs_OLS_test.pdf", format = 'pdf',bbox_inches=None, dpi=300)

temp_xarr = cf.arr_to_xarr(np.log10(np.ma.masked_array(regression_sum,GRS_masked.mask)),GRS_lats,GRS_lons,"volc_1_surf")
# temp_xarr = cf.arr_to_xarr(regression_sum,GRS_lats,GRS_lons,"volc_1_surf")

gdf = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/best_reg_volcs_test.gpkg',driver="GPKG")


# All volcs correlation
all_corr_sum = np.zeros([36,72])
bad_volcs = ["Peneus_Patera","Amphritites","Malea_Patera"]
no_AC_volcs = set(df_volc['Volcano Name'].tolist()) ^ set(bad_volcs)
regressor_OLS_all_volc = sm.OLS(GRS_vals_flattened_trans,sm.add_constant(volc_1_dict_trans_df[no_AC_volcs])).fit()

params = regressor_OLS_all_volc.params
# params[params<=0] = 0
for element, volc_name in enumerate(no_AC_volcs):
    temp = volc_1_dict[volc_name] * params[element+1]
    all_corr_sum += temp

all_corr_sum += params[0]


fig4,axs4 = plt.subplots(2,1,sharex=True,sharey=True)
axs4=axs4.flatten()

im=axs4[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(all_corr_sum,GRS_masked.mask),norm=colors.LogNorm(),edgecolors='none')
# im=axs4[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(all_corr_sum,GRS_masked.mask),norm=colors.SymLogNorm(linthresh=1e10,vmin=np.min(all_corr_sum),vmax=np.max(all_corr_sum)),edgecolors='none')
axs4[0].set_title("OLS all sum. r^2 = .307")
axs4[0].grid(True)


fig4.colorbar(im,ax=axs4[0],orientation='vertical')

im=axs4[1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs4[1].set_title("GRS")
axs4[1].grid(True)
fig4.colorbar(im,ax=axs4[1],orientation='vertical')


fig4.savefig(out_dir+"all_volcs_OLS_test.pdf", format = 'pdf',bbox_inches=None, dpi=300)


temp_xarr = cf.arr_to_xarr(np.log10(np.ma.masked_array(all_corr_sum,GRS_masked.mask)),GRS_lats,GRS_lons,"volc_1_surf")
gdf = cf.xr_to_geodf(temp_xarr,"ESRI:104971")
gdf.to_file('/Users/tylerpaladino/Documents/ISU/Thesis/Mars_regolith_hydration/LMD_MATHAM_Utils/data/all_volcs_test.gpkg',driver="GPKG")

r_all_volc = cf.get_r_val(all_corr_sum,GRS_vals_flattened_trans)
fig7,axs7 = plt.subplots(2,1,sharex=True,sharey=True)
axs7=axs7.flatten()

im=axs7[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(all_corr_sum,GRS_masked.mask),norm=colors.LogNorm(),edgecolors='none')
axs7[0].set_title("OLS Regression all volc sum. r^2 = .307, AIC = -257.5 ")
axs7[0].grid(True)
axs7[0].scatter(df_volc['lon'],df_volc['lat'],c='red',marker='^')

# for i,txt in enumerate(DA_corr_volcs):
#     axs3[0].annotate(txt,(DA_corr_locs['lon'][i],DA_corr_locs['lat'][i]))
    
fig7.colorbar(im,ax=axs7[0],orientation='vertical')

im=axs7[1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs7[1].set_title("GRS")
axs7[1].grid(True)
fig7.colorbar(im,ax=axs7[1],orientation='vertical')

fig7.savefig(out_dir+"all_volcs_OLS.pdf", format = 'pdf',bbox_inches=None, dpi=300)
# Binary biserial correlation
BS_corr_volcs = ["Apollinaris_Patera","Elysium_Mons","Cerberus",
                   "Syrtis_Major","Alba_Patera"]
BS_corr_locs = df_volc[df_volc["Volcano Name"].isin(BS_corr_volcs)].reset_index()
BS_corr_sum = np.zeros([36,72])
for volc_name in BS_corr_volcs:
    temp = volc_1_dict[volc_name]
    temp2 = cf.Cont2Dich(temp,5e10)
    BS_corr_sum += temp
    BS_corr_sum[BS_corr_sum>1] = 1
    
fig6,axs6 = plt.subplots(2,1,sharex=True,sharey=True)
axs6=axs6.flatten()

im=axs6[0].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(BS_corr_sum,GRS_masked.mask),edgecolors='none')
axs6[0].set_title("Pearson corr BS volc sum. r = .437. Threshold = 5e10")
axs6[0].grid(True)
axs6[0].scatter(BS_corr_locs['lon'],BS_corr_locs['lat'],c='red',marker='^')

# for i,txt in enumerate(DA_corr_volcs):
#     axs3[0].annotate(txt,(DA_corr_locs['lon'][i],DA_corr_locs['lat'][i]))
    
fig6.colorbar(im,ax=axs6[0],orientation='vertical')

im=axs6[1].pcolormesh(GRS_lons,GRS_lats,GRS_grid)
axs6[1].set_title("GRS")
axs6[1].grid(True)
fig6.colorbar(im,ax=axs6[1],orientation='vertical')

fig8,axs8 = plt.subplots()

# best_AIC = pd.read_csv('Best_AIC.csv')
# best_AIC.plot(x='Variable',y='Coefficient',kind='bar',yerr='SE',capsize=4,rot=45,ax=axs8)
# axs8.grid(False)
# axs8.set_title('Relative Importance')
# axs8.set_ylabel('OLS Coefficient')

# fig8.savefig(out_dir+"OLS_relative_importance.pdf", format = 'pdf',bbox_inches=None, dpi=300)





# axs9.grid(False)
# axs9.set_title('Relative Importance')
# axs9.set_ylabel('OLS Coefficient')

# fig9.savefig(out_dir+"OLS_relative_importance.pdf", format = 'pdf',bbox_inches=None, dpi=300)


pp = sns.pairplot(data=final_volc_1_df,y_vars=['GRS'],x_vars=df_volc['Volcano Name'].tolist())
for ax in pp.axes.flat:
    ax.set(xscale="log")

all_volc_sum = np.zeros([36,72])
for volc_name in p_set[-1]:
    temp = volc_1_dict[volc_name]
    all_volc_sum += temp


volc1_log = np.log10(volc_1_dict["Hadriaca_Patera"])
log_min = float(volc1_log.min())
log_max = float(volc1_log.max())
plfig = px.imshow(volc_1_dict["Hadriaca_Patera"], animation_frame='Time')
plfig.show()

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
volc_interp = interp((GRS_lats,GRS_lons))
plt.pcolormesh(GRS_lons,GRS_lats,volc_interp)



# plt.pcolormesh(lon,lat, volc_sum)


print("pause")

volc_merged = xr.merge([volc1_sum,volc2_sum,volc3_sum,volc4_sum])

volc_merged.to_netcdf('all_volc_dep_norm.nc')

# volc1_sum.to_netcdf('all_volc1_dep.nc')
# volc2_sum.to_netcdf('all_volc2_dep.nc')
# volc3_sum.to_netcdf('all_volc3_dep.nc')
# volc4_sum.to_netcdf('all_volc4_dep.nc')