# deap_experiments

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
cp -r Stephen/pendulum ~/scratch/$exp_code
cd ~/scratch/$exp_code
```
run 5 parallel experiments
```
for seed in `seq 1 5`; do sbatch ./ccanada.sh -s $seed; done
```
