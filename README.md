# LMD_MATHAM_Utils

This repository contains all necessary scripts, input files, and data files for recreating the analysis done in Paladino et al. (2023). It is recommended to create a conda environment first using the provided `environment.txt` file:

`conda env create -name env_name --file environment.txt`

Note that this work depends on two separate models: the Generic PCM and ATHAM. The Generic PCM can be downloaded and installed from [here](http://www-planets.lmd.jussieu.fr/). I've made fairly significant edits to parts of the Generic PCM to allow volcanic ash tracer tracking as well as just playing friendly with ATHAM. This modified code is available at this separate [GitHub repository](https://github.com/palatyle/LMD_GCM_Paladino). I would recommend installing the Generic PCM first and making sure that it works, then making edits to the codes indicated in the readme of my GitHub repo linked above. 

ATHAM must be requested from one of the authors, Michael Herzog. I've since made some small quality of life edits to ATHAM (referred to now as MATHAM - Mars ATHAM) as well as edits to make it work on Mars (also small). If you have ATHAM installed and end up trying to recreate what I've done, please contact me and I'd be happy to help.

The general order of operations for this work are:

1. Run base Generic PCM simulation with no active volcanism occuring (`callvolcano=.false.` in `callphys.def`)
2. `GCM2MATHAM.py` - Extracts ATHAM-friendly atmospheric profiles from the GCM output
3. `GCM_MATHAM_create.py` - Creates all necessary files and directories to run chained ATHAM-PCM simulations
4. `batch_start.sh` - Queues all simulations prepped in step 3 into the HPC system
5. `LMD_interp.py` - Interpolates output from step 4 to GRS grid.
6. Analysis on interpolated data

## Scripts used in Paladino et al. (2023)

`utils/GCM2MATHAM.py` -- Script to generate atmospheric profiles from the Generic PCM* netcdf output over every major volcano found in `data/Mars_Volc_locs.csv`. This only needs be run once per base atmosphere.

`utils/GCM_MATHAM_create.py` -- Script to generate individual directories containing everything needed to run both the Generic PCM and the martian adaptation to ATHAM (MATHAM). You will need to edit the first couple lines of code indicating where the various data, model, and I/O directories are.

`utils/LMD_interp.py` -- Script to interpolate model output from the PCM to the resolution and extent of the GRS H<sub>2</sub>O data. This script will loop through all output from the coupled PCM-MATHAM models and output geopackages.

`utils/MATHAM_domain_flux.py` -- Script to sum up ash in MATHAM. Outputs a text file with each column indicating a different ash tracer as well as height above the ground. This script is callable from the command line.

`utils/Sensitivity_setup.py` -- Essentially the same as `utils/GCM_MATHAM_create.py`, but creates directories for sensitivity tests. 

`utils/Sensitivity_tests_viz.py` -- script to visualize the output from sensitivity tests.

`utils/Surface_p_comp.py` -- Compares surface pressure of ancient vs modern Mars. Creates Figure 3.  

`utils/batch_start.sh` -- shell script to be placed in the same directory as the generated volcano directories. Will loop through each volcano folder and begin the model runs. 

`utils/common_funcs.py` -- Common functions used throughout other scripts

`utils/nc_volc_filt.py` -- script used to filter out extraneous data from the PCM output not relevant to this project. Also compresses netcdf file. Results in massive storage savings.

`utils/pbs_chain_template.sh`  -- Shell script template to chain together multiple pbs jobs. In this case, we chain together `MATHAM` -> `domain_flux.pbs` -> `PCM` -> `nc_volc_filt.py`. Referenced in `utils/batch_start.sh`

\* Note that the name of the Generic PCM was recently changed from "LMD GCM" to "Generic PCM". As such, many references in the code/code names still retain this original naming convention.

## PBS scripts
`pbs_files/MATHAM_pbs.pbs` -- PBS template for running MATHAM. Used in `utils/GCM_MATHAM_create.py`

`pbs_files/domain_flux.pbs` -- PBS template for the domain flux code (`utils/MATHAM_domain_flux.py`). Used in `utils/GCM_MATHAM_create.py`.

`pbs_files/nc_volc_filt.pbs` -- PBS template for the netcdf output filtering code (`utils/nc_volc_filt.py`). Used in `utils/GCM_MATHAM_create.py`.

`pbs_files/pbs_LMD.pbs` -- PBS template for running the Generic PCM. Used in `utils/GCM_MATHAM_create.py`


## Data files

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8114833.svg)](https://doi.org/10.5281/zenodo.8114833) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8114972.svg)](https://doi.org/10.5281/zenodo.8114972) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8115051.svg)](https://doi.org/10.5281/zenodo.8115051) -- Set 1-3 of compressed PCM netCDF output for each volcano hosted on Zenodo. Used in `utils/LMD_interp.py`

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8102490.svg)](https://doi.org/10.5281/zenodo.8102490) --Sensitivty tests netCDF output hosted on Zenodo. Used in `utils/Sensitivity_tests_viz.py`


`data/DCI_raster_reproj.tif` -- Dust Cover index raster originally sourced from Ruff and Christensen (2002) and reprojected.

`data/GRS_data_S_180.csv` -- CSV containing GRS Sulfur data with the convention changed to -180 to 180. Data originally from Rani et al. (2022).

`data/GRS_data_raw_180.csv` -- CSV containing GRS H<sub>2</sub>O data with the convention changed to -180 to 180. Data originally from Rani et al. (2022).

`data/Mars_Volc_locs.csv` -- CSV containing all major potentially explosive volcanoes on Mars as well as their lat/lon coordinates. Originally from Kerber et al. (2012) and subsequently augmented.

`data/no_tharsis.` -- this directory mostly contains the interpolated ash deposition maps for the analysis done in the paper. Each of these outputs is separated by volcano location as well as tracer. They are in `.npy` format and can be read in easily with numpy (`np.load(file.npy)`)

In this directory, you'll also find two matlab binary files (`.mat`). `PCM_geo.mat` contains geometry information, while `ps_tharsis_no_tharsis.mat` contains surface pressure information. Both of these files are used in the `Surface_p_comp.py` script. 
## Model Inputs
    P_MATHAM/ 
    ├── IO_ref/ -- Contains MATHAM input files generated in `GCM2MATHAM.py`
    │   └── input/` -- Contains INPUT_kinetic
    │
    cold_dry_no_tharsis/ -- Contains inputs for the PCM (*.def and start files) as well as scripts for compiling the PCM as well a various PCM helper programs.  

## Outputs

`GWR_output_summary.xlsx` -- Excel file containing major output from GWR analysis done in Arc as well as some plots.

`GWR_project.ppkx` -- ArcGIS Pro Project Package containing raw and transformed GCM output as well as all GWR analyses. 
