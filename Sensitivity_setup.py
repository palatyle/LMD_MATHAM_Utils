import glob
import os
import shutil

main_GCM_dir = '/home/palatyle/LMD_gen/trunk/cold_dry_no_tharsis/'
MATHAM_dir = '/home/palatyle/P_MATHAM/'
code_dir = '/home/palatyle/LMD_MATHAM_Utils/'
GCM_datadir = '/home/palatyle/LMD_gen/trunk/datadir'

# Volcano names filename
volc_fn = '/home/palatyle/GCM2MATHAM/Mars_Volc_locs.csv'

# Volcano to test on
volc_name = "Apollinaris_Patera"
atmos = "cold_dry"

# Different sensitivity test names
keys = ['low_temp','high_temp','low_H2O','high_H2O','med_rho','high_rho','high_MER']
for keyword in keys:
    # Make directory for each sensitivity test and cd into it
    os.mkdir(keyword)
    os.chdir(keyword)

    # Create directory of volcano name and cd into it
    os.mkdir(volc_name)
    os.chdir(volc_name)

    # Create directory in scratch in the format of keyword_VolcanoName_AtmosphereType
    os.mkdir("/scratch/palatyle/" + keyword + '_' + volc_name + "_" + atmos)
    os.mkdir(atmos)
    os.chdir(atmos)
    current_dir = os.getcwd()

    # Copy over all .def files from master GCM directory
    for file in glob.glob(main_GCM_dir+'*.def'):
        shutil.copy2(os.path.join(main_GCM_dir,file),current_dir)

    # Edit callphys.def file 
    file = open('callphys.def','r')
    callphys_def = file.readlines()
    
    # Edit output dir line
    callphys_def[21] = "output_dir = /scratch/palatyle/" + keyword + '_' + volc_name + "_" + atmos + "/" + "diagfi.nc\n"
    callphys_def[24] = "callvolcano=.true."
    
    # Edit volcano name line
    callphys_def[46] = "volc_name=" + volc_name + "\n"

    # Edit atmosphere type line
    callphys_def[51] = "atmos_type=" + atmos + "\n"

    # Edit volcanic ash density line
    if keyword == 'med_rho':
        density = 1500
    elif keyword == 'high_rho':
        density = 2500
    else:
        density = 700
    callphys_def[54] = "rho_volc=" + str(density) + "\n"

    # Edit input_key line (controls other sensitivities)
    callphys_def[57] = "input_key = " + keyword + "\n"

    # Write edited lines to file
    file = open('callphys.def','w')
    file.writelines(callphys_def)
    file.close()


    # Copy over GCM executable 
    shutil.copy2(os.path.join(main_GCM_dir,'gcm_128x96x23_phystd_para.e'),current_dir)

    # Copy over .dat file
    shutil.copy2(os.path.join(main_GCM_dir,'Bands_128x96x23_48prc.dat'),current_dir)

    # Copy over LMD pbs file
    shutil.copy2(os.path.join(main_GCM_dir,'pbs_LMD.pbs'),current_dir)

    # Edit LMD pbs file
    file = open('pbs_LMD.pbs','r')
    LMD_pbs = file.readlines()
    # Edit -N flag in pbs file
    LMD_pbs[4] = "#PBS -N LMD_" + volc_name + "_" + atmos + "_" + keyword + "\n"
    # Edit mpirun line
    LMD_pbs[19] = "mpirun " + current_dir + "/gcm_128x96x23_phystd_para.e\n"

    file = open("pbs_LMD.pbs","w")
    file.writelines(LMD_pbs)
    file.close()


    # Copy over start files from GCM
    shutil.copy2(os.path.join(main_GCM_dir,'restart.nc'),current_dir+"/start.nc")
    shutil.copy2(os.path.join(main_GCM_dir,'restartfi.nc'),current_dir+"/startfi.nc")

    # Copy MATHAM pbs file for each season
    for season in ["winter","spring","summer","fall"]:
        shutil.copy2(os.path.join(MATHAM_dir,'Mars_test_warm_wet.pbs'),current_dir+"/MATHAM_"+ volc_name + "_" + atmos + "_" + season + ".pbs")
        file = open("MATHAM_"+ volc_name + "_" + atmos + "_" + season + ".pbs","r")
        MATHAM_pbs = file.readlines()
        
        MATHAM_pbs[7] = "#PBS -N MATHAM_" + volc_name + "_" + atmos + "_" + season + "_" + keyword + "\n"

        MATHAM_exec = "/home/palatyle/P_MATHAM/exec/atham"
        
        i_flag = " -i /home/palatyle/P_MATHAM/IO_ref"
        o_flag = " -o /scratch/palatyle/" + keyword + '_' + volc_name + "_" + atmos
        f_flag = " -f MATHAM_" + season
        a_flag = " -a INPUT_atham_setup_Mars"
        p_flag = " -p " + volc_name + "_" + season + "_" + atmos + "_no_tharsis"
        v_flag = " -v INPUT_volcano_mars_" + keyword
        d_flag = " -d INPUT_dynamic_setup"

        MATHAM_pbs[33] = "mpirun " + MATHAM_exec + i_flag + o_flag + f_flag + a_flag + p_flag + v_flag + d_flag + "\n"
        file = open("MATHAM_"+ volc_name + "_" + atmos + "_" + season + ".pbs","w")
        file.writelines(MATHAM_pbs)
        file.close()

        # Domain flux pbs edit
        shutil.copy2(os.path.join(code_dir,'domain_flux.pbs'),current_dir+"/domain_flux_"+ volc_name + "_" + atmos + "_" + season + ".pbs")
        file = open("domain_flux_"+ volc_name + "_" + atmos + "_" + season + ".pbs","r")
        domain_flux_pbs = file.readlines()

        domain_flux_pbs[4] = "#PBS -N domain_flux_" + volc_name + "_" + atmos + "_" + season + "_" + keyword + "\n"
        domain_flux_pbs[18] = "python " + code_dir + "MATHAM_domain_flux.py -i /scratch/palatyle/" + keyword + '_' + volc_name + "_" + atmos + "/MATHAM_"+season+"_netCDF_MOV.nc -o " + GCM_datadir + '/' + volc_name + "_" + season + "_" + atmos + "_" + keyword + ".txt\n"
        
        file = open("domain_flux_"+ volc_name + "_" + atmos + "_" + season + ".pbs","w")
        file.writelines(domain_flux_pbs)
        file.close()

        # nc volc filt pbs edit. Hacky if statement to only od this once isntead of for every season
        if season =="winter":
            os.mkdir("input")
            shutil.copy2(MATHAM_dir +'/input/INPUT_kinetic',current_dir+"/input/INPUT_kinetic")

            shutil.copy2(os.path.join(code_dir,'nc_volc_filt.pbs'),current_dir+"/nc_volc_filt_"+ volc_name + "_" + atmos + ".pbs")
            file = open("nc_volc_filt_"+ volc_name + "_" + atmos + ".pbs","r")
            nc_volc_filt_pbs = file.readlines()

            nc_volc_filt_pbs[4] = "#PBS -N nc_volc_filt_" + volc_name + "_" + atmos + "_" + keyword + "\n"
            nc_volc_filt_pbs[18] = "python " + code_dir +  "nc_volc_filt.py -i /scratch/palatyle/" + keyword + '_' + volc_name + "_" + atmos + "/" + "diagfi.nc -o /scratch/palatyle/" + keyword + '_' + volc_name + "_" + atmos + "/" + volc_name+ "_diagfi_volc_filt.nc\n"
            
            file = open("nc_volc_filt_"+ volc_name + "_" + atmos + ".pbs","w")
            file.writelines(nc_volc_filt_pbs)
            file.close()

        # Pbs chain edits
        if season == "winter":
            shutil.copy2(os.path.join(code_dir,'pbs_chain_template.sh'),current_dir+"/pbs_chain_"+ volc_name + "_" + atmos + ".sh")
            file = open("pbs_chain_"+ volc_name + "_" + atmos + ".sh","r")
            pbs_chain = file.readlines()
            pbs_chain[3] = season+"=$(qsub " + "MATHAM_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
            pbs_chain[8] = season+"_flux=$(qsub -W depend=afterok:$" + season + " domain_flux_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
        elif season == "spring":
            pbs_chain[4] = season+"=$(qsub " + "MATHAM_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
            pbs_chain[9] = season+"_flux=$(qsub -W depend=afterok:$" + season + " domain_flux_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
        elif season == "summer":
            pbs_chain[5] = season+"=$(qsub " + "MATHAM_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
            pbs_chain[10] = season+"_flux=$(qsub -W depend=afterok:$" + season + " domain_flux_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
        elif season == "fall":
            pbs_chain[2] = season+"=$(qsub " + "MATHAM_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
            pbs_chain[7] = season+"_flux=$(qsub -W depend=afterok:$" + season + " domain_flux_" + volc_name + "_" + atmos + "_" + season + ".pbs)\n"
            pbs_chain[12] = "LMD=$(qsub -W depend=afterok:$fall_flux:$winter_flux:$spring_flux:$summer_flux pbs_LMD.pbs)\n"
            pbs_chain[14] = "filt=$(qsub -W depend=afterok:$LMD nc_volc_filt_" + volc_name + "_" + atmos + ".pbs)\n"
            file = open("pbs_chain_"+ volc_name + "_" + atmos + ".sh","w")
            file.writelines(pbs_chain)
            file.close()
            os.system("chmod +x "+"pbs_chain_"+ volc_name + "_" + atmos + ".sh")

    print(volc_name + " done!")
    os.chdir('../../../')

            