#!/bin/bash

fall=$(qsub MATHAM_Hadriaca_Patera_cold_dry_fall.pbs)
winter=$(qsub MATHAM_Hadriaca_Patera_cold_dry_winter.pbs)
spring=$(qsub MATHAM_Hadriaca_Patera_cold_dry_spring.pbs)
summer=$(qsub MATHAM_Hadriaca_Patera_cold_dry_summer.pbs)

fall_flux=$(qsub -W depend=afterok:$fall domain_flux_Hadriaca_Patera_cold_dry_fall.pbs)
winter_flux=$(qsub -W depend=afterok:$winter domain_flux_Hadriaca_Patera_cold_dry_winter.pbs)
spring_flux=$(qsub -W depend=afterok:$spring domain_flux_Hadriaca_Patera_cold_dry_spring.pbs)
summer_flux=$(qsub -W depend=afterok:$summer domain_flux_Hadriaca_Patera_cold_dry_summer.pbs)

LMD=$(qsub -w depend=afterok:$fall_flux:$winter_flux:$spring_flux:$summer_flux pbs_LMD.pbs)

filt=$(qsub -w depend=afterok:$LMD nc_volc_filt_.pbs)