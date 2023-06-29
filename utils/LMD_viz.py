import os

import common_funcs as cf
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.sans-serif':'Myriad Pro'})
plt.rcParams.update({'xtick.labelsize': 16, 
                     'ytick.labelsize': 16,
                     'axes.titlesize': 14,
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

# Read in GRS data
df_GRS = pd.read_csv('GRS_data_raw_180.csv')

GRS_lats,GRS_lons,GRS_vals_flattened,GRS_grid = cf.GRS_wrangle(df_GRS)

GRS_masked = np.ma.masked_invalid(GRS_grid)

os.chdir('no_tharsis')

volc_1_dict = {}

max_vals = []
min_vals = []
for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    volc_1_dict[volc_name]= np.load(volc_name + '_volc_1_surf'  + '.npy')
    max_vals.append(volc_1_dict[volc_name].max())
    min_vals.append(volc_1_dict[volc_name].min())

# log plot pcolor
fig1,axs1 = plt.subplots(5,4,sharex=True,sharey=True)
axs1=axs1.flatten()

for cnt,volc_name in enumerate(df_volc['Volcano Name']):
    im=axs1[cnt].pcolormesh(GRS_lons,GRS_lats,np.ma.masked_array(volc_1_dict[volc_name],GRS_masked.mask),norm=colors.LogNorm(),edgecolors='none')
    axs1[cnt].set_title(volc_name)
    axs1[cnt].grid(True)
    im.set_clim(min(min_vals),max(max_vals))

fig1.colorbar(im,ax=axs1.ravel().tolist(),label='Ash Area Density (kg/m^2)')
fig1.suptitle('Deposited Ash for Each Volcano')
fig1.supxlabel('Longitude')
fig1.supylabel('Latitude')
fig1.savefig(out_dir+"all_volcs_log_pcolor.pdf", format = 'pdf',bbox_inches=None, dpi=300)

