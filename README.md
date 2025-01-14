# covid_restimulation_model

Analysis script:
compute_SPIKE_TOL_raw_data_hyperOpt_LogLikelihoodModel_SAVE_ENTROPIES.py

Job submission scripts:
run_compute_raw_data_hyperOpt_LogLikelihoodModel.sh -> submit_MLE.sh


run_compute_raw_data_hyperOpt_LogLikelihoodModel.sh:
  - allows you to iterate over the 9-mer aggregation functions, submitting
    separate scripts for each.

submit_MLE.sh:
  - runs each individual job
