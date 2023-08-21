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

module load python
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip
pip install -r requirements_alliancecan.txt

python -u ccanada_main.py $seed > main-$seed-$SLURM_JOB_ID.std