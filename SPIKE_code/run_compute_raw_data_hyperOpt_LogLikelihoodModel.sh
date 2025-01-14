# For cluster

#PBS -l nodes=1:ppn=1
echo "hello"

cd /mnt/scratch/marcust/WorkingDir/Cansu_COVID/SPIKE_code
export PYTHONPATH=/mnt/scratch/marcust/WorkingDir/CFIT/cfit:$PYTHONPATH

source /home/marcust/anaconda3/bin/activate general_3.9

randomizeHLAs=False
use_conc=True


# outdir='/mnt/scratch/marcust/WorkingDir/Cansu_COVID/MLE_Results/Updated/pool_stim_pool_restim/model_noTol_noVacc_CpSpec_inclCrossRx_inclConc'
# use_tolerance=False
# include_vaccine=False
# Cp_specific=True
# include_crossReactivity=True
# max_evals=12000 #20000 #600 #

# outdir='/mnt/scratch/marcust/WorkingDir/Cansu_COVID/MLE_Results/Updated/pool_stim_pool_restim/model_noTol_noVacc_CpSpec_noCrossRx_inclConc'
# use_tolerance=False
# include_vaccine=False
# Cp_specific=True
# include_crossReactivity=False
# max_evals=12000 #20000 #600 #

# outdir='/mnt/scratch/marcust/WorkingDir/Cansu_COVID/MLE_Results/Updated/pool_stim_pool_restim/model_noTol_inclVacc_CpSpec_inclCrossRx_inclConc'
# use_tolerance=False
# include_vaccine=True
# Cp_specific=True
# include_crossReactivity=True
# max_evals=12000 #20000 #600 #

outdir='/mnt/scratch/marcust/WorkingDir/Cansu_COVID/MLE_Results/Updated/pool_stim_pool_restim/Entropy_Perplexity_Results'
use_tolerance=False
include_vaccine=False
Cp_specific=True
include_crossReactivity=True
max_evals=1


for aggrFun in max  #sum mean max min realSoftMax realSoftMin gmean
do
  for HLA_reshuffle_number in 0 #11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30  #0 #1 2 3 4 5 6 7 8 9 10 #set to 0 if randomizeHLAs=False
  do
    # # -q standard.q@atletico.conifold
    # # spike model that includes tolerance
    # job1Name=${aggrFun}_reshff_${HLA_reshuffle_number}_withTol
    # job1Name=${aggrFun}_without_MUTMUT_MUTWT_nocrossRx

    # IEDB="Everything"
    # job1Name=${aggrFun}_with_all_IEDB
    # qsub -q standard.q@acmilan.conifold -N $job1Name ./submit_MLE.sh "$aggrFun" "$max_evals" "$use_tolerance" "$include_vaccine" "$Cp_specific" "$include_crossReactivity" "$use_conc" "$outdir" "$HLA_reshuffle_number" "$randomizeHLAs" "$IEDB"

    # IEDB="No_Covid"
    # job1Name=${aggrFun}_no_Covid
    # qsub -q standard.q@atletico.conifold -N $job1Name ./submit_MLE.sh "$aggrFun" "$max_evals" "$use_tolerance" "$include_vaccine" "$Cp_specific" "$include_crossReactivity" "$use_conc" "$outdir" "$HLA_reshuffle_number" "$randomizeHLAs" "$IEDB"

    # IEDB="Only_Covid"
    # job1Name=${aggrFun}_only_Covid
    # qsub -q standard.q@bayern.conifold -N $job1Name ./submit_MLE.sh "$aggrFun" "$max_evals" "$use_tolerance" "$include_vaccine" "$Cp_specific" "$include_crossReactivity" "$use_conc" "$outdir" "$HLA_reshuffle_number" "$randomizeHLAs" "$IEDB"


    IEDB="No_IEDB"
    job1Name=${aggrFun}__${HLA_reshuffle_number}_Save_Entropy_Plots
    qsub -q standard.q@cska.conifold -N $job1Name ./submit_MLE.sh "$aggrFun" "$max_evals" "$use_tolerance" "$include_vaccine" "$Cp_specific" "$include_crossReactivity" "$use_conc" "$outdir" "$HLA_reshuffle_number" "$randomizeHLAs" "$IEDB"

    echo "submittted: aggrFun " $aggrFun
    # sleep 130
    sleep 3


  done
done
