# For cluster

#PBS -l nodes=1:ppn=1
echo "hello"
source /home/marcust/anaconda3/bin/activate general_3.9

cd /mnt/scratch/marcust/WorkingDir/Cansu_COVID/SPIKE_code
# python3 compute_SPIKE_TOL_raw_data_hyperOpt_LogLikelihoodModel.py -ninemerFun $1 -max_evals $2 -use_tolerance $3 -include_vaccine $4 -Cp_specific $5 -include_crossReactivity $6 -use_conc $7 -outdir $8 -HLA_reshuffle_number $9 -randomizeHLAs ${10} -IEDB ${11}
python3 compute_SPIKE_TOL_raw_data_hyperOpt_LogLikelihoodModel_SAVE_ENTROPIES.py -ninemerFun $1 -max_evals $2 -use_tolerance $3 -include_vaccine $4 -Cp_specific $5 -include_crossReactivity $6 -use_conc $7 -outdir $8 -HLA_reshuffle_number $9 -randomizeHLAs ${10} -IEDB ${11}
