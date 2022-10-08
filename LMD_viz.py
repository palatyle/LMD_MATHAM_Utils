import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('D:\\')

df = pd.read_csv('Mars_Volc_locs.csv')


for volc_name in df['Volcano Name']:
    ds = xr.open_dataset('diagfi_volc_filt_nccomp_'+volc_name+'.nc',mask_and_scale=False,decode_times=False)
    volc1 = ds.volc_1_surf[-1,:,:]
    lat = ds.latitude
    lon = ds.longitude
    plt.pcolormesh(lon,lat, volc1)
    plt.show()
    print("pause")
