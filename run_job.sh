#! /bin/bash
set -e

source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate gengli_env

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

python_file=$1 
# Removes args and resets to $1
shift 

# python "$python_file" --band "$2" --whitening_tw 10 --observation_tw 4 --clean_output_filename "clean_statistical_results_obs_4s" --glitch_output_filename "glitch_statistical_results_obs_4s"
python "$python_file" "$@"