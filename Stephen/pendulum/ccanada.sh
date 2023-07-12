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
source /home/skelly/deap_env/bin/activate
python -u memory_01.py $seed > memory_01-$seed.std #`date +%Y-%m-%d-%H-%M-%S`.std