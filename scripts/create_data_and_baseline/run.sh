#!/bin/bash

export PYTHONUNBUFFERED=1;

printf "\nRunning full pipeline for directory: %s " "$1";
printf "\n";
stdbuf -i0 -o0 -e0 python 00_get_observing_conditions.py --base-directory "$1";
printf "\n";
stdbuf -i0 -o0 -e0 python 01_plot_observing_conditions.py --base-directory "$1";
printf "\n";
stdbuf -i0 -o0 -e0 python 02_create_data_sets.py --base-directory "$1";
printf "\n";
stdbuf -i0 -o0 -e0 python 03_run_pca.py --base-directory "$1";
printf "\n";
stdbuf -i0 -o0 -e0 python 04_compute_pca_snrs.py --base-directory "$1";
printf "\n";
stdbuf -i0 -o0 -e0 python 05_plot_pca_snr_over_npc.py --base-directory "$1";
printf "\n";
stdbuf -i0 -o0 -e0 python 06_run_median_adi_and_compute_snr.py --base-directory "$1";
printf "\n";
printf "Completed pipeline for directory: %s \n\n" "$1";
