from nco import Nco
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