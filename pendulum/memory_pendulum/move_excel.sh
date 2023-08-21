#!/bin/bash

dir=$(pwd)

cd ../deap_experiments/pendulum/work_on_memory
git rm memory_raw_data.xlsx

cd $dir
cp memory_raw_data.xlsx ../deap_experiments/pendulum

cd ../deap_experiments/pendulum

git pull
git add .
git commit -m "add excel file"
git push

git mv memory_raw_data.xlsx ./work_on_memory
git commit -m "move excel file"
git push