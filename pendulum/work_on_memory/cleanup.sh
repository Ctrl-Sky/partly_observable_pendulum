#!/bin/bash
if ls *.std 1> /dev/null 2>&1; then rm *.std; fi
if ls slurm* 1> /dev/null 2>&1; then rm slurm*; fi
if ls *.png 1> /dev/null 2>&1; then rm *.png; fi
