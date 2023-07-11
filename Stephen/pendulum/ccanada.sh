#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --time=3:00:00
#SBATCH --mem=0 
#SBATCH --ntasks=48 
#SBATCH --nodes=1

module load python
source /home/skelly/deap_env/bin/activate
python -u memory_01.py > memory_01-`date +%Y-%m-%d-%H-%M-%S`.std