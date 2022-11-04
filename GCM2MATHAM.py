import netCDF4 as nc
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


def import_netcdf_obj(filename):
    '''
    Imports the netcdf object

    Input: filename
    Outputs: netcdf data object
    '''
    return nc.Dataset(filename)



def find_closest(arr, arr_num):
    '''
    Finds the index in arr closest to the values specified in arr_num

    Parameters
    -----------
    arr: full array
    arr_num: single num

    Returns
    -----------

    '''

    arr_diff = abs(arr - arr_num)

    # min index of of arr_diff
    arr_idx = np.where(arr_diff == min(arr_diff))

    # convert to int
    return int(arr_idx[0])

def season_selector(nc_ds,sol_lon,arr_name,full_arr):
    '''
    Inputs an array and splits it up into 4 seprate arrays based on season. 

    Parameters
    -----------
    nc_ds: netcdf dataset object
    sol_lon: Ls value array for entire simulation
    arr_name: string of array to be read in
    full_arr: Boolean contorolling whether array being read in is 4D array or 3D

    Returns
    -----------
    arr_spring, arr_summer, arr_fall, arr_winter: 3 or 2D arrays containing values for one season
    
    '''
    summer_ls_idx = find_closest(sol_lon, 90)
    fall_ls_idx = find_closest(sol_lon, 180)
    winter_ls_idx = find_closest(sol_lon, 270)

    try:
        if full_arr == True:
            arr_spring = nc_ds[arr_name][0,:,:,:]
            arr_summer = nc_ds[arr_name][summer_ls_idx,:,:,:]
            arr_fall = nc_ds[arr_name][fall_ls_idx,:,:,:]
            arr_winter = nc_ds[arr_name][winter_ls_idx,:,:,:]  
        else:
            arr_spring = nc_ds[arr_name][0,:,:]
            arr_summer = nc_ds[arr_name][summer_ls_idx,:,:]
            arr_fall = nc_ds[arr_name][fall_ls_idx,:,:]
            arr_winter = nc_ds[arr_name][winter_ls_idx,:,:]  
    except(IndexError):
            arr_spring = arr_summer = arr_fall = arr_winter = np.zeros_like(nc_ds['temp'][0,:,:,:])
            
    return arr_spring, arr_summer, arr_fall, arr_winter



def profile_creation(arr, lat_idx, lon_idx, **kwargs):
    '''
    Inputs an array and pulls out a vertical profile at some lat,lon coordinate. If surface array exists, uses that as the bottom point. Otherwise, just uses 0

    Parameters
    -----------
    arr: array to be turned into profile
    lat_idx: index of latitude pt
    lon_idx: index of longitude pt
    kwargs = surf_arr: Reads in surface array if present.

    Returns
    -----------
    profile: vertical profile at lat, lon point. 
    
    '''
    surf_arr = kwargs.get('surf_arr', None)
    if surf_arr is not None:
        profile = np.append(surf_arr[lat_idx,lon_idx],arr[:,lat_idx,lon_idx])
    else:
        profile = np.append(0,arr[:,lat_idx,lon_idx])
    return profile

def write_profiles(profile, prof_name, directory):
    
    '''
    Writes profile array to file in correct format to play friendly with FORTRAN code

    Parameters
    -----------
    profile: 2D array to be written  
    prof_name: string representing profile name to be written
    directory: string representing directory to put profile in

    Returns
    -----------
    None
    
    '''
    filename = directory + prof_name
    with open(filename,'w') as fout:
        fout.write('{:17}    {}\n'.format('NUMBER OF MARPTS:',profile.shape[1]))
        fout.write('\n')
        fout.write(' {:7}  {:13} {:14} {:7} {:7}\n'.format('HEIGHTS','TEMPERATURES','REL. HUMIDITY','U-WIND','V-WIND'))
        fout.write('{:>8} {:>13} {:>14} {:>7} {:>7}\n'.format('[M]','[C]','[%]','[M/S]','[M/S]'))
        fout.write('\n')

        for line in np.transpose(profile):
            fout.write('{:8.1f}      {:8.2f}      {:9.5f}  {:6.2f}  {:6.2f}'.format(*line))
            fout.write('\n')


# Datafile

for atmos in ["cold_dry"]: #,"warm_wet"]:
    if atmos == "cold_dry":
        dir = '/home/palatyle/LMD_gen/trunk/cold_dry_h2o/'
    elif atmos == "warm_wet":
        dir = '/home/palatyle/LMD_gen/trunk/warm_wet/'
    fn = dir+'diagfi.nc'

    # Volcano names filename
    volc_fn = '/home/palatyle/GCM2MATHAM/Mars_Volc_locs_Arabia.csv'

    volc_df = pd.read_csv(volc_fn)




    out = '/home/palatyle/P_MATHAM/IO_ref/'

    # Lat,lon coords of volcano of interest
    # lat_volc = 30.0
    # lon_volc = -30.0


    # volc_name = "temp_test_"

    print("Reading netcdf file")
    # import netcdf object and remove automasks for all variables
    ds = import_netcdf_obj(fn)
    for k in ds.variables:
        ds.variables[k].set_auto_mask(False)

    # Import lat, lon, and altitude
    lat = ds['latitude'][:] # Degrees
    lon = ds['longitude'][:] # Degrees
    alt = ds['altitude'][:] *1000 # m

    # Import time and solar longitude arrays
    time = ds['Time'][:]
    ls = ds['Ls'][:] # Solar longitude

    for index, row in volc_df.iterrows():
        print("Processing "+ row['Volcano Name'])
        lat_volc = row['lat']
        lon_volc = row['lon']
        volc_name = row['Volcano Name']

        # Find indics of volcano lat, lon coords
        lat_volc_idx = find_closest(lat,lat_volc)
        lon_volc_idx = find_closest(lon,lon_volc)

        # Import and split up temperature, pressure, relative humidity and u/v wind vector arrays based on season
        temp_spr,temp_sum,temp_fall,temp_win = np.subtract(season_selector(ds, ls, 'temp', True), 273.15) # K
        tsurf_spr,tsurf_sum,tsurf_fall,tsurf_win = np.subtract(season_selector(ds, ls, 'tsurf', False), 273.15) # K surf
        pres_spr,pres_sum,pres_fall,pres_win = season_selector(ds, ls, 'p',True) # Pa
        psurf_spr,psurf_sum,psurf_fall,psurf_win = season_selector(ds, ls, 'ps', False) # Pa surf
        RH_spr,RH_sum,RH_fall,RH_win = season_selector(ds, ls, 'RH',True) # Relative Humidity [0-1]
        RH_spr,RH_sum,RH_fall,RH_win = RH_spr*100,RH_sum*100,RH_fall*100,RH_win*100 # Relative Humidity [%]
        u_spr,u_sum,u_fall,u_win = season_selector(ds, ls, 'u',True) # m/s
        v_spr,v_sum,v_fall,v_win = season_selector(ds, ls, 'v',True) # m/s

        # Create individual profiles of each meterological parameter

        alt_profile = np.append(0,alt)

        print("Create profiles")
        # Spring
        temp_prof_spr = profile_creation(temp_spr, lat_volc_idx, lon_volc_idx, surf_arr=tsurf_spr)
        pres_prof_spr = profile_creation(pres_spr, lat_volc_idx, lon_volc_idx, surf_arr=psurf_spr)
        RH_prof_spr = profile_creation(RH_spr, lat_volc_idx, lon_volc_idx)
        u_prof_spr = profile_creation(u_spr, lat_volc_idx, lon_volc_idx)
        v_prof_spr = profile_creation(v_spr, lat_volc_idx, lon_volc_idx)

        spr_profiles = np.vstack((alt_profile, temp_prof_spr, RH_prof_spr,u_prof_spr,v_prof_spr))

        # Summer
        temp_prof_sum = profile_creation(temp_sum, lat_volc_idx, lon_volc_idx, surf_arr=tsurf_sum)
        pres_prof_sum = profile_creation(pres_sum, lat_volc_idx, lon_volc_idx, surf_arr=psurf_sum)
        RH_prof_sum = profile_creation(RH_sum, lat_volc_idx, lon_volc_idx)
        u_prof_sum = profile_creation(u_sum, lat_volc_idx, lon_volc_idx)
        v_prof_sum = profile_creation(v_sum, lat_volc_idx, lon_volc_idx)

        sum_profiles = np.vstack((alt_profile, temp_prof_sum, RH_prof_sum,u_prof_sum,v_prof_sum))


        # Fall
        temp_prof_fall = profile_creation(temp_fall, lat_volc_idx, lon_volc_idx, surf_arr=tsurf_fall)
        pres_prof_fall = profile_creation(pres_fall, lat_volc_idx, lon_volc_idx, surf_arr=psurf_fall)
        RH_prof_fall = profile_creation(RH_fall, lat_volc_idx, lon_volc_idx)
        u_prof_fall = profile_creation(u_fall, lat_volc_idx, lon_volc_idx)
        v_prof_fall = profile_creation(v_fall, lat_volc_idx, lon_volc_idx)

        fall_profiles = np.vstack((alt_profile, temp_prof_fall, RH_prof_fall,u_prof_fall,v_prof_fall))


        # Winter
        temp_prof_win = profile_creation(temp_win, lat_volc_idx, lon_volc_idx, surf_arr=tsurf_win)
        pres_prof_win = profile_creation(pres_win, lat_volc_idx, lon_volc_idx, surf_arr=psurf_win)
        RH_prof_win = profile_creation(RH_win, lat_volc_idx, lon_volc_idx)
        u_prof_win = profile_creation(u_win, lat_volc_idx, lon_volc_idx)
        v_prof_win = profile_creation(v_win, lat_volc_idx, lon_volc_idx)

        win_profiles = np.vstack((alt_profile[:], temp_prof_win[:], RH_prof_win[:],u_prof_win[:],v_prof_win[:]))

        print("write data")
        write_profiles(spr_profiles, volc_name+'_spring_'+ atmos, out)
        write_profiles(sum_profiles, volc_name+'_summer_'+ atmos, out)
        write_profiles(fall_profiles, volc_name+'_fall_'+ atmos, out)
        write_profiles(win_profiles, volc_name+'_winter_'+ atmos, out)

        # fig, ax = plt.subplots()

        # ax.set_xticks(lon)
        # ax.set_yticks(lat)

        # for i in range(tsurf.shape[0]):
        #     plt.imshow(tsurf[i,:,:], extent=[lon.min(), lon.max(), lat.min(), lat.max()])
        #     plt.plot(lon[50],lat[25],'o')
        #     plt.colorbar()
        #     plt.show()
            

        print('Done')
