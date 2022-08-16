import netCDF4 as nc
import numpy as np
import argparse

def import_netcdf_obj(filename):
    '''
    Input: filename
    Outputs: netcdf data object
    '''
    return nc.Dataset(filename)

def import_geometry(ds):
    '''
    Returns x, y, z dimensions from netcdf file. 
    '''
    xvar = m2km(ds.variables['x'][:])
    yvar = m2km(ds.variables['y'][:])
    zvar = m2km(ds.variables['z'][:])

    return xvar, yvar, zvar 

def m2km(dim_length):
    '''
    Returns input m length value to km
    '''
    return dim_length/1000

def get_timestep_val(ds):
    return ds.variables['time'][:]

def import_tracer(ds,tracer_name):
    
    tracer_nc = ds.variables[tracer_name]
    tracer_np = np.array(tracer_nc)
    tracer_np[tracer_np == tracer_nc._FillValue] = 0
    
    return tracer_np


def row_sum_ts(ash_var):
    """_summary_

    Args:
        ash_var (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    domain_exit1 = np.squeeze(ash_var[:,0,:]) # side
    domain_exit2 = np.squeeze(ash_var[:,-1,:]) # side
    domain_exit3 = np.squeeze(ash_var[:,:,0]) # side
    domain_exit4 = np.squeeze(ash_var[:,:,-1]) # side
    domain_exit5 = np.squeeze(ash_var[-1,:,:]) # top
    

    # sum row wise 
    dom1_sum = np.sum(domain_exit1,axis=1)
    dom2_sum = np.sum(domain_exit2,axis=1)
    dom3_sum = np.sum(domain_exit3,axis=1)
    dom4_sum = np.sum(domain_exit4,axis=1)
    dom5_sum = np.sum(domain_exit5) #except here, just sum the whole shebang 
    
   
    row_sum = dom1_sum + dom2_sum + dom3_sum + dom4_sum
    row_sum[-1] = row_sum[-1] + dom5_sum # add in top domain to last row
    return row_sum

def sum_remaining_ash(ash_var):
    """_summary_

    Args:
        ash_var (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return np.sum(np.sum(ash_var,axis=1),axis=1)
    


def domain_flux_calc(ash_var1,ash_var2,ash_var3,ash_var4,z_var,time_array,fname):
    """_summary_

    Args:
        ash_var1 (_type_): _description_
        ash_var2 (_type_): _description_
        ash_var3 (_type_): _description_
        ash_var4 (_type_): _description_
        z_var (_type_): _description_
        time_array (_type_): _description_
    """    
    ash1_row_ts = []
    ash2_row_ts = []
    ash3_row_ts = []
    ash4_row_ts = []
    for time_idx,time in enumerate(time_array):
        if time_idx != len(time_arr)-1:
            ash1_row_ts.append(row_sum_ts(ash_var1[time_idx,:,:,:]))
            ash2_row_ts.append(row_sum_ts(ash_var2[time_idx,:,:,:]))
            ash3_row_ts.append(row_sum_ts(ash_var3[time_idx,:,:,:]))
            ash4_row_ts.append(row_sum_ts(ash_var4[time_idx,:,:,:]))


    # Convert list to array
    ash1_row_ts = np.vstack(ash1_row_ts)
    ash2_row_ts = np.vstack(ash2_row_ts)
    ash3_row_ts = np.vstack(ash3_row_ts)
    ash4_row_ts = np.vstack(ash4_row_ts)

    total_ash1_ts = np.sum(ash1_row_ts,axis=0)
    total_ash2_ts = np.sum(ash2_row_ts,axis=0)
    total_ash3_ts = np.sum(ash3_row_ts,axis=0)
    total_ash4_ts = np.sum(ash4_row_ts,axis=0)

    ash1_domain = sum_remaining_ash(ash_var1[-1,:,:,:])
    ash2_domain = sum_remaining_ash(ash_var2[-1,:,:,:])
    ash3_domain = sum_remaining_ash(ash_var3[-1,:,:,:])
    ash4_domain = sum_remaining_ash(ash_var4[-1,:,:,:])

    total_domain_flux_ash1 = total_ash1_ts + ash1_domain
    total_domain_flux_ash2 = total_ash2_ts + ash2_domain
    total_domain_flux_ash3 = total_ash3_ts + ash3_domain
    total_domain_flux_ash4 = total_ash4_ts + ash4_domain

    final_write_arr = np.column_stack((z_var,total_domain_flux_ash1,total_domain_flux_ash2,total_domain_flux_ash3,total_domain_flux_ash4))
    np.savetxt(fname,final_write_arr,fmt='%.8e',delimiter=',',header='Height (km), 7_8 um ash conc (g/kg), 15_6 um ash conc (g/kg), 125 um ash conc (g/kg), 1 mm ash conc (g/kg)',comments='')
    
    
parser=argparse.ArgumentParser()
parser.add_argument('--input_file','-i',help='Full path to netcdf file')
parser.add_argument('--output_file','-o',help='Full path to output file')
args = parser.parse_args()

fn = args.input_file
out_name = args.output_file

# fn = '/Users/tylerpaladino/Documents/ISU/Thesis/ATHAM_wind/polar_127_5m_300ms_50ms/atham_netCDF_MOV.nc'

# get netcdf object 
netcdf_obj = import_netcdf_obj(fn)
# get x y and z vectors
x,y,z = import_geometry(netcdf_obj)

time_arr = get_timestep_val(netcdf_obj)


# read in entire ash and density tracers
ash1 = import_tracer(netcdf_obj, 'ash1')
ash2 = import_tracer(netcdf_obj, 'ash2')
ash3 = import_tracer(netcdf_obj, 'ash3')
ash4 = import_tracer(netcdf_obj, 'ash4')

# out_name = 

domain_flux_calc(ash1,ash2,ash3,ash4,z,time_arr,out_name)