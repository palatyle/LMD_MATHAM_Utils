import pandas as pd
import os
import subprocess as sp

volc_fn = '/home/palatyle/GCM2MATHAM/Mars_Volc_locs.csv'

volc_df = pd.read_csv(volc_fn)

for volc_name in volc_df['Volcano Name']:
    os.chdir(volc_name)
    for atmos in ["cold_dry","warm_wet"]:
        os.chdir(atmos)
        print("Submitting "+volc_name+"_"+atmos)
        sp.run("qsub pbs_LMD.pbs",shell=True)
        os.chdir("..")
    os.chdir("..")

