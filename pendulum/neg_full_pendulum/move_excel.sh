#!/bin/bash

dir=$(pwd)
data="neg_full_raw_data.xlsx"
fld="neg_full_pendulum

cd ../deap_experiments/pendulum/$fld
git rm $data

cd $dir
cp $data ../deap_experiments/pendulum

cd ../deap_experiments/pendulum

git pull
git add .
git commit -m "add excel file"
git push

git mv $data ./$fld
git commit -m "move excel file"
git push