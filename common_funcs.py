from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pandas as pd
import geopandas
from itertools import chain, combinations
import numpy as np
from scipy.interpolate import griddata 
from scipy.stats import shapiro, pearsonr, boxcox


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

def transformation_test(arr):
    lambdas = np.linspace(-1,1,5)
    p_vals = []
    sp_vals = []
    for la in lambdas:
        sp,p = shapiro(boxcox(arr,la))
        p_vals.append(p)
        sp_vals.append(sp)
    return sp_vals[np.argmax(p_vals)], np.max(p_vals), lambdas[np.argmax(p_vals)]



def get_r_val(volc_dep,GRS_dat):
    # Reshape to flat array
    volc_dep_flat = volc_dep.flatten()
    # Remove nans from both arrays
    volc_dep_flat_nn = volc_dep_flat[~np.isnan(GRS_dat)]
    GRS_dat_flat_nn = GRS_dat[~np.isnan(GRS_dat)]
    
    sp_val, sp_p_val, max_lambda = transformation_test(volc_dep_flat_nn)
    
    sp = (sp_val,sp_p_val)
    
    r = pearsonr(GRS_dat_flat_nn,boxcox(volc_dep_flat_nn,max_lambda))
 
    return r, sp, max_lambda

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
    data_flat = data_grd.flatten()
    
    return lats,lons,data_flat,data_grd