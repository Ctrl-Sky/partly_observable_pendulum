#!/bin/bash

dir=$(pwd)

cd ../deap_experiments/pendulum/part_obs_pendulum
git rm part_obs_raw_data.xlsx

cd $dir
cp part_obs_raw_data.xlsx ../deap_experiments/pendulum

cd ../deap_experiments/pendulum

git pull
git add .
git commit -m "add excel file"
git push

git mv part_obs_raw_data.xlsx ./part_obs_pendulum
git commit -m "move excel file"
git push