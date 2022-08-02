import pandas as pd
import os 
import shutil
import glob
master_GCM_dir = '/home/palatyle/LMD_gen/trunk/test_run/'
cold_dry_dir = '/home/palatyle/LMD_gen/trunk/cold_dry/'
warm_wet_dir = '/home/palatyle/LMD_gen/trunk/warm_wet/'
MATHAM_dir = '/home/palatyle/m-atham/'
# Volcano names filename
volc_fn = '/home/palatyle/GCM2MATHAM/Mars_Volc_locs.csv'

volc_df = pd.read_csv(volc_fn)

for volc_name in volc_df['Volcano Name']:

    os.mkdir(volc_name)

    os.chdir(volc_name)

    for atmos in ["cold_dry","warm_wet"]:
        os.mkdir(atmos)
        os.mkdir("/scratch/palatyle/" + volc_name + "_" + atmos)
        os.chdir(atmos)
        current_dir = os.getcwd()

        # Copy over all .def files from master GCM directory
        for file in glob.glob(master_GCM_dir+'*.def'):
            shutil.copy2(os.path.join(master_GCM_dir,file),current_dir)

        # Edit callphys.def file 
        file = open('callphys.def','r')
        callphys_def = file.readlines()
        # Edit output dir line
        callphys_def[21] = "output_dir = /scratch/palatyle/" + volc_name + "_" + atmos + "/" + "diagfi.nc\n"
        # Edit volcano name line
        callphys_def[46] = "volc_name="+volc_name+"\n"

        # Edit atmosphere type line
        if atmos == "cold_dry":
            callphys_def[51] = "atmos_type=cd\n"
        elif atmos == "warm_wet":
            callphys_def[51] = "atmos_type=ww\n"

        # Write edited lines to file
        file = open('callphys.def','w')
        file.writelines(callphys_def)
        file.close()


        # Copy over GCM executable 
        shutil.copy2(os.path.join(master_GCM_dir,'gcm_128x96x23_phystd_para.e'),current_dir)

        # Copy over .dat file
        shutil.copy2(os.path.join(master_GCM_dir,'Bands_128x96x23_48prc.dat'),current_dir)
        
        # Copy over LMD pbs file
        shutil.copy2(os.path.join(master_GCM_dir,'pbs_LMD.pbs'),current_dir)
        
        # Edit LMD pbs file
        file = open('pbs_LMD.pbs','r')
        LMD_pbs = file.readlines()
        # Edit -N flag in pbs file
        LMD_pbs[4] = "#PBS -N LMD_" + volc_name + "_" + atmos + "\n"
        # Edit mpirun line
        LMD_pbs[19] = "mpirun " + current_dir + "/gcm_128x96x23_phystd_para.e\n"

        file = open("pbs_LMD.pbs","w")
        file.writelines(LMD_pbs)
        file.close()

        # Copy over start files depending on relevant atmospheric scenario
        if atmos == "cold_dry":
            shutil.copy2(os.path.join(cold_dry_dir,'start.nc'),current_dir)
            shutil.copy2(os.path.join(cold_dry_dir,'startfi.nc'),current_dir)
        elif atmos == "warm_wet":
            shutil.copy2(os.path.join(warm_wet_dir,'start.nc'),current_dir)
            shutil.copy2(os.path.join(warm_wet_dir,'startfi.nc'),current_dir)

        # MATHAM things
        # Copy over executable
        # Copy over pbs file
        # Copy over matlab script? 

        # Copy MATHAM pbs file for each season
        for season in ["winter","spring","summer","fall"]:
            shutil.copy2(os.path.join(MATHAM_dir,'MATHAM_cold_dry_wint_st.pbs'),current_dir+"/MATHAM_"+ volc_name + "_" + atmos + "_" + season + ".pbs")
            file = open("MATHAM_"+ volc_name + "_" + atmos + "_" + season + ".pbs","r")
            MATHAM_pbs = file.readlines()
            
            MATHAM_pbs[5] = "#PBS -N MATHAM_" + volc_name + "_" + atmos + "_" + season + "\n"

            MATHAM_exec = "/home/palatyle/m-atham/exec/atham"
            i_flag = " -i /home/palatyle/m-atham/IO_ref"
            o_flag = " -o /scratch/palatyle/" + volc_name + "_" + atmos
            f_flag = " -f MATHAM_" + season
            a_flag = " -a INPUT_matham_setup_MATHAM_cold_dry"
            if atmos == "cold_dry":
                p_flag = " -p "+ volc_name + "_" + season + "_cd" 
            elif atmos == "warm_wet":
                p_flag = " -p "+ volc_name + "_" + season + "_ww"
            
            v_flag = " -v INPUT_volcano_MATHAM_phreato_low_MER"
            d_flag = " -d INPUT_dynamic_setup"

            MATHAM_pbs[27] = "mpirun " + MATHAM_exec + i_flag + o_flag + f_flag + a_flag + p_flag + v_flag + d_flag + "\n"
            file = open("MATHAM_"+ volc_name + "_" + atmos + "_" + season + ".pbs","w")
            file.writelines(MATHAM_pbs)
            file.close()
        
        os.chdir('..')
    print(volc_name + " done!")
    os.chdir('..')

        