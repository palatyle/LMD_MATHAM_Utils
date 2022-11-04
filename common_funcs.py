from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pandas as pd
import geopandas
from itertools import chain, combinations
import numpy as np
from scipy.interpolate import griddata 
import scipy.stats


def normalize(arr):
     return arr/np.max(arr)

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

def interpolate_data(X,Y,Xi,Yi,arr):
    # Set up interpolator object which has original grid data
    interpolator = RegularGridInterpolator((X,Y),arr,method="nearest")
    # Interpolate to Xi,Yi grid
    return interpolator((Xi,Yi))

def xr_to_geodf(xr_dat,crs):
    df = xr_dat.to_dataframe()
    df = df.reset_index()
    geodf = geopandas.GeoDataFrame(df,geometry=geopandas.points_from_xy(df.longitude,df.latitude))
    geodf = geodf.set_crs(crs) 
    return geodf

def arr_to_xarr(arr, lats_arr, lons_arr, name):
    return xr.DataArray(data = arr,coords={'latitude': lats_arr,'longitude': lons_arr},dims=['latitude','longitude'],name=name)

def interpolate_data(X,Y,Xi,Yi,arr):
    # Set up interpolator object which has original grid data
    interpolator = RegularGridInterpolator((X,Y),arr,method="nearest")
    # Interpolate to Xi,Yi grid
    return interpolator((Xi,Yi))

def get_r_val(arr,arr_nans_flat):
    # Reshape to flat array
    arr_flat = arr.reshape(arr.size)
    # Remove nans from both arrays
    arr_flat_nn = arr_flat[~np.isnan(arr_nans_flat)]
    arr_nans_flat_nn = arr_nans_flat[~np.isnan(arr_nans_flat)]
    
    r = scipy.stats.pearsonr(arr_nans_flat_nn,arr_flat_nn)
    sp = scipy.stats.shapiro(arr_flat_nn)
    return r, sp

def GRS_wrangle(df):
    # Sort data and drop Center Lon column
    df_sorted = df.sort_values(by=['lon_180','CenterLat'],ascending=False)
    df_sorted = df_sorted.drop(columns=['CenterLon'])

    # Convert to numpy array and only grab unique vals
    lons = df_sorted['lon_180'].to_numpy()
    lons = np.unique(lons)

    lats = df_sorted['CenterLat'].to_numpy()
    lats = np.unique(lats)
    lats = np.sort(lats)
    lats = lats[::-1]
    
    # Create GRS meshgrid
    lon_grid,lat_grid = np.meshgrid(lons,lats)

    # Interpolate (really just reshape) GRS data to meshgrid
    data_grd = griddata((df['lon_180'].to_numpy(),df['CenterLat'].to_numpy()),df['Concentration'].to_numpy(),(lon_grid,lat_grid),method='nearest')
    data_flat = data_grd.reshape(data_grd.size)
    
    return lats,lons,data_flat