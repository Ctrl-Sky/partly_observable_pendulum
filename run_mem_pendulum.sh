#!/bin/bash

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python pendulum/memory_pendulum/mem_pendulum.py