# Partly Observable Pendulum Task

## Welcome

Hi! My name is Sky Quan and welcome to my repository. This repo was entirely created during my first co-op as a research assistant working under Dr. Stephen Kelly from May 2023 to August 2023. We worked towards creating evolving adaptable control policies with mental models in partially observable and dynamic environments through the use of Gymnasium's pendulum task and Distributed Evolutionary Algorithms in Python's (DEAP) genetic programming library. To accomplish this task, we redesigned the pendulum to be partly observable by removing the pendulum's angular velocity from its observation space. We then solved the partly observable task using two methods, a recursive window that saved and added the pendulum's previous positions to its observation space and an indexed memory approach that provided the agent with a dynamic array that the agent could read and write to at will. If you are interested in the implementation I highlight the rec_pendulum.py and mem_pendulum.py file for the recursive and indexed memory method respectively. (Note: Many algorithms were designed to run using Compute Canada, an outside server, however, I specifically designed those files to run on a local machine). Using the successful agents from the partly observable test, we looked to design one that could work in any environment facing any gravity value. Unfortunately due to the end of my co-op, the work here is still unfinished, however, I still invite you to look around and contact me if you have any questions!

## Notes on running parallel experiments on alliance clusters

ssh login
```
ssh username@cedar.computacanada.ca
```
clone repository (always work in scratch filesystem)
```
cd ~/scratch
git clone https://gitlab.cas.mcmaster.ca/kellys32/deap_experiments.git
```
update repository
```
cd ~/scratch/deap_experiments
git pull origin main
```
copy code to uniq experiment directory (from deap_experiments)
```
exp_code=deap-pendulum-po-`date +%Y-%m-%d-%H-%M-%S`-`git rev-parse --short HEAD`
cp -r pendulum/neg_men_pendulum ~/scratch/$exp_code
cp -r pendulum/modules ~/scratch
cd ~/scratch/$exp_code
```
run 5 parallel experiments
```
for seed in `seq 1 13`; do sbatch ./ccanada.sh -s $seed; done
```
command to check running experiments
```
squeue -u <usrname>
```
to copy experiment folder to local computer, first get full path from inside exp directory:
```
pwd
```
...then on local computer, use scp to copy remote directory to local computer:
```
scp -r cedar.computecanada.ca:/path-to-experiment ./
```
