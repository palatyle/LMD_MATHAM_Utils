from nco import Nco
import xarray as xr
import argparse

nco = Nco()

parser=argparse.ArgumentParser()
parser.add_argument('--input_file','-i',help='Full path to netcdf file')
parser.add_argument('--output_file','-o',help='Full path to output file')
args = parser.parse_args()


fn = args.input_file
out_name = args.output_file


# Extracts only volcano vars (and necessary dependent geo vars) from netcdf file.
nco.ncks(input=fn,output=out_name,options=["-v \"volc_*\""])


# Reread in and compress
ds = xr.open_dataset(out_name,mask_and_scale=False,decode_times=False)

comp = dict(zlib=True, complevel=1)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(out_name+'comp.nc', encoding=encoding)