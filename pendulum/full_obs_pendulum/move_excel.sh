#!/bin/bash

dir=$(pwd)

cd ../deap_experiments/pendulum/full_obs_pendulum
git rm full_obs_raw_data.xlsx

cd $dir
cp full_obs_raw_data.xlsx ../deap_experiments/pendulum

cd ../deap_experiments/pendulum

git pull
git add .
git commit -m "add excel file"
git push

git mv memory_raw_data.xlsx ./full_obs_pendulum
git commit -m "move excel file"
git push