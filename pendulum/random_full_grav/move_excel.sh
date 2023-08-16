#!/bin/bash

dir=$(pwd)

cd ../deap_experiments/pendulum/random_full_grav
git rm random_full_obs_raw_data.xlsx

cd $dir
cp random_full_obs_raw_data.xlsx ../deap_experiments/pendulum

cd ../deap_experiments/pendulum

git pull
git add .
git commit -m "add excel file"
git push

git mv random_full_obs_raw_data.xlsx ./random_full_grav
git commit -m "move excel file"
git push