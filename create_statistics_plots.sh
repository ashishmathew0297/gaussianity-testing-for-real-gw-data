#!/bin/bash

# python statistics_results.py \
#     --clean_output_file "./outputs/clean_4.0s_statistical_results_full.csv" \
#     --glitch_output_file "./outputs/glitch_4.0s_statistical_results_full.csv" \
#     --latex_output_file "./outputs/test_conf_matrix_full_4s.tex" \
#     --csv_output_file "./outputs/test_conf_matrix_full_4s.csv" \
#     --image_postfix "4.0s_full"

# python statistics_results.py \
#     --clean_output_file "./outputs/clean_4.0s_statistical_results_high.csv" \
#     --glitch_output_file "./outputs/glitch_4.0s_statistical_results_high.csv" \
#     --latex_output_file "./outputs/test_conf_matrix_high_freq_4s.tex" \
#     --csv_output_file "./outputs/test_conf_matrix_high_freq_4s.csv" \
#     --image_postfix "4.0s_high_freq"

# python statistics_results.py \
#     --clean_output_file "./outputs/clean_4.0s_statistical_results_low.csv" \
#     --glitch_output_file "./outputs/glitch_4.0s_statistical_results_low.csv" \
#     --latex_output_file "./outputs/test_conf_matrix_low_freq_4s.tex" \
#     --csv_output_file "./outputs/test_conf_matrix_low_freq_4s.csv" \
#     --image_postfix "4.0s_low_freq"

# create directories for new outputs if thet don't exist

latex_output_dir="./outputs_new/conf_latex"
csv_output_dir="./outputs_new/conf_csv"

mkdir -p ./outputs_new/conf_latex
mkdir -p ./outputs_new/conf_csv

for scaler in "standard" "minmax" "robust"
do

python statistics_results.py \
    --clean_output_file "./outputs_new/clean_1.0s_${scaler}_statistical_results_full.csv" \
    --glitch_output_file "./outputs_new/glitch_1.0s_${scaler}_statistical_results_full.csv" \
    --latex_output_file "${latex_output_dir}/test_conf_matrix_${scaler}_full_1s.tex" \
    --csv_output_file "${csv_output_dir}/test_conf_matrix_${scaler}_full_1s.csv" \
    --image_postfix "1.0s_${scaler}_full"

python statistics_results.py \
    --clean_output_file "./outputs_new/clean_1.0s_${scaler}_statistical_results_low.csv" \
    --glitch_output_file "./outputs_new/glitch_1.0s_${scaler}_statistical_results_low.csv" \
    --latex_output_file "${latex_output_dir}/test_conf_matrix_${scaler}_low_freq_1s.tex" \
    --csv_output_file "${csv_output_dir}/test_conf_matrix_${scaler}_low_freq_1s.csv" \
    --image_postfix "1.0s_${scaler}_low_freq"

python statistics_results.py \
    --clean_output_file "./outputs_new/clean_1.0s_${scaler}_statistical_results_high.csv" \
    --glitch_output_file "./outputs_new/glitch_1.0s_${scaler}_statistical_results_high.csv" \
    --latex_output_file "${latex_output_dir}/test_conf_matrix_${scaler}_high_freq_1s.tex" \
    --csv_output_file "${csv_output_dir}/test_conf_matrix_${scaler}_high_freq_1s.csv" \
    --image_postfix "1.0s_${scaler}_high_freq"
done