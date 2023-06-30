import scipy.io as sio
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib as mpl
import numpy as np

plt.rcParams.update({'font.sans-serif':'Myriad Pro',
                     'font.stretch': 'condensed'})
plt.rcParams.update({'xtick.labelsize': 16, 
                     'ytick.labelsize': 16,
                     'axes.titlesize': 20,
                     'figure.titlesize': 16,
                     'axes.labelsize': 16,
                     'axes.labelsize': 16,
                     'legend.fontsize': 14,
                     'legend.title_fontsize': 14,      
                     'figure.facecolor':(240/255,240/255,240/255),
                     'savefig.facecolor':(240/255,240/255,240/255),})

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
mpl.rc('axes.formatter', use_mathtext=True)
# Change directory to where this script is
os.chdir(os.path.dirname(__file__))

# Go up to data/no_tharsis directory 
os.chdir('../data/no_tharsis')

# Load in PCM geometry lists (lat/lon)
geo = sio.loadmat('PCM_geo.mat',simplify_cells=True)

# Load in PCM pressure arrays for Tharsis and no Tharsis scenarios
Ps_dict = sio.loadmat('ps_tharsis_no_tharsis.mat',simplify_cells=True)

# Create Figure
fig,ax = plt.subplots(2,1,figsize=(9,6),sharex=True,sharey=False)

# Plot modern topography pressure
im = ax[0].pcolormesh(geo['lon'],geo['lat'],np.transpose(Ps_dict['ps_tharsis']),cmap='viridis',rasterized=True)
ax[0].set_title('Modern Topography', weight=700)
ax[0].grid(True,color='w',linewidth=.5,alpha=0.4)
ax[0].set_ylabel('Latitude',labelpad=0)
ax[0].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))

# Find and set min/max values for colorbar
im.set_clim(min(Ps_dict['ps_tharsis'].min(),Ps_dict['ps_no_tharsis'].min()),max(Ps_dict['ps_tharsis'].max(),Ps_dict['ps_no_tharsis'].max()))

# Plot ancient topography pressure
im = ax[1].pcolormesh(geo['lon'],geo['lat'],np.transpose(Ps_dict['ps_no_tharsis']),cmap='viridis',rasterized=True)
ax[1].set_title('Ancient Topography (No Tharsis Rise)', weight=700, stretch = 'condensed')
ax[1].grid(True,color='w',linewidth=.5,alpha=0.4)
ax[1].set_xlabel('Longitude', labelpad=0)
ax[1].set_ylabel('Latitude', labelpad=0)
ax[1].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
ax[1].xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
ax[1].xaxis.set_major_locator(plt.MaxNLocator(7))

# Find and set min/max values for colorbar
im.set_clim(min(Ps_dict['ps_tharsis'].min(),Ps_dict['ps_no_tharsis'].min()),max(Ps_dict['ps_tharsis'].max(),Ps_dict['ps_no_tharsis'].max()))

fig.tight_layout()

fig.colorbar(im,ax=ax.ravel().tolist(),label='Surface Pressure (Pa)')
fig.savefig('../../Ps_diff.pdf',dpi=300,format='pdf',bbox_inches='tight')