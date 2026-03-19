#!/bin/bash
#------------------
#SBATCH --account=gratis
#------------------
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=epyc2
#SBATCH --qos=job_debug

module load Python/3.10.4-GCCcore-11.3.0
source /storage/homefs/kd23t352/dev/deep-graph-matching/venv/bin/activate

python3 "$@"

deactivate
module unload Python/3.10.4-GCCcore-11.3.0
