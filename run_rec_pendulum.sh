#!/bin/bash

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python pendulum/part_obs_pendulum/rec_pendulum.py