# LMD_MATHAM_Utils

## Scripts used in Paladino et al. (2023)

`GCM2MATHAM.py` -- Script to generate atmospheric profiles from the Generic PCM* netcdf output over every major volcano found in data/Mars_Volc_locs.csv. This only needs be run once per base atmosphere.

`GCM_MATHAM_create.py` -- Script to generate individual directories containing everything needed to run both the Generic PCM and the martian adaptation to ATHAM (P_MATHAM). You will need to edit the first couple lines of code indicating where the PCM, PCM data, and MATHAM directories are. 

`LMD_interp.py` -- Script to interpolate model output from the PCM to the resolution and extent of the GRS H<sub>2</sub>O data. This script will loop through all output from the coupled PCM-MATHAM mopdels and output geopackages. 

`MATHAM_domain_flux.py` -- Script to sum up ash in MATHAM. Outputs a text file with each column indicating a different ash tracer as well as height above the ground. This script is callable from the command line.

`Sensitivity_setup.py` -- Essentially the same as GCM_MATHAM_create.py, but creates directories for sensitivity tests. 

`Sensitivity_tests_viz.py` -- scripts to visualize the output from sensitivity tests.

`batch_start.sh` -- shell script to be placed in the same directory as the generated volcano directories. Will loop through each volcano folder and begin the model runs. 

`common_funcs.py` -- Common functions used throughout the other scripts

`domain_flux.pbs` -- PBS template for the domain flux code (`MATHAM_domain_flux.py`). Used in `GCM_MATHAM_create.py`.

`nc_volc_filt.pbs` -- PBS template for the netcdf output filtering code (`nc_volc_filt.py`). Used in `GCM_MATHAM_create.py`.

`nc_volc_filt.py` -- script used to filter out extraneous data from the PCM output not relevant to this project. Also compresses netcdf file. Results in massive storage savings.

`pbs_chain_template.sh`  -- Shell script template to chain together multiple pbs jobs. In this case, we chain together `MATHAM` -> `domain_flux.pbs` -> `PCM` -> `nc_volc_filt.py`. Used in `batch_start.sh`

`P_MATHAM/MATHAM_pbs.pbs` -- PBS template for running MATHAM. Used in `GCM_MATHAM_create.py`

`cold_dry_no_tharsis/pbs_LMD.pbs` -- PBS template for running the Generic PCM. Used in `GCM_MATHAM_create.py`

\* Note that the name of the Generic PCM was recently changed from "LMD GCM" to "Generic PCM". As such, many references in the code/code names still retain this original naming convention.


## Data files

`data/DCI_raster_reproj.tif` -- Dust Cover index raster originally sourced from Ruff and Christensen (2002) and reprojected.

`data/GRS_data_S_180.csv` -- CSV containing GRS Sulfur data with the convention changed to -180 to 180. Data originally from Rani et al., (2022).

`data/GRS_data_raw_180.csv` -- CSV containing GRS H<sub>2</sub>O data with the convention changed to -180 to 180. Data originally from Rani et al., (2022).

`data/Mars_Volc_locs.csv` -- CSV containing all major potentially explosive volcanoes on Mars as well as their lat/lon coordinates. 

## Model Inputs
    P_MATHAM/ 
    ├── IO_ref/ -- Contains MATHAM input files generated in `GCM2MATHAM.py`
    │   └── input/` -- Contains INPUT_kinetic
    │
    cold_dry_no_tharsis/ -- Contains inputs for the PCM (*.def and start files) as well as scripts for compiling various PCM helper programs.  
