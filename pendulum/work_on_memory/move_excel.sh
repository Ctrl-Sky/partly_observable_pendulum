#!/bin/bash

dir=$(pwd)

cd ../deap_experiments/pendulum/work_on_memory
rm memory_raw_data.xlsx

cd $dir
cp memory_raw_data.xlsx ../deap_experiments/pendulum/work_on_memory

git add
git commit -m "add excel file"
git push