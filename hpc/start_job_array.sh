#!/bin/bash
#SBATCH --array=0-524
#SBATCH --time=0-01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# etc.


module load Python/3.10.4-GCCcore-11.3.0
source "."

DATASET="IMDB-MULTI"
NUM_JOBS=295
PAIRS_PER_JOB=3000
TOTAL_PAIRS=883785

START=$((SLURM_ARRAY_TASK_ID * PAIRS_PER_JOB))
END=$((START + PAIRS_PER_JOB))

# last job takes remainder
if [ $END -gt $TOTAL_PAIRS ]; then
    END=$TOTAL_PAIRS
fi

echo "Job $SLURM_ARRAY_TASK_ID processing pairs $START to $((END - 1))"

python3 calc_ged_arrays.py \
	--dataset_dir data/TUDataset \
	--dataset_name $DATASET \
	--output_dir res/$DATASET \
	--pairs_file pairs/${DATASET}_pairs.npy \
	--metadata pairs/${DATASET}_metadata.json \
	--timeout 1 \
	--start_idx $START \
	--end_idx $END

deactivate
module unload Python/3.10.4-GCCcore-11.3.0
