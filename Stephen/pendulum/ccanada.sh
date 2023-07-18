#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --time=3:00:00
#SBATCH --mem=0 
#SBATCH --ntasks=48 
#SBATCH --nodes=1

seed=1

while getopts s: flag
do
   case "${flag}" in
      s) seed=${OPTARG};;
   esac
done

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ../requirements_alliancecan.txt

python -u memory_01.py $seed > memory_01-$seed-$SLURM_JOB_ID.std