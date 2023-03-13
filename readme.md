# Collection of scripts for Si-based grain boundary simulations

This repo is used for synchronizing calculation setups and post-processing. 

## Contains:
1. `pipelines.py`
Pipelines for various 'classes' of calculations. Currently tested only `geo_opt` for geometry optimization and `freq` for frequency calculations. Current code is *not flexible*. This will change as we encounter more diverse structures. 
2. `nl.py`
for calculating neighbor lists. 
3. `plot_charges.py`
for visualizing bader charges. WIP merge into `pipelines.py`

## Usage:
install into your environment:
```bash
# installation
cd && mkdir scripts 
conda install -n base -c conda-forge conda-build
conda activate <env_name>
git clone git@github.com:xyttyxy/sigb_scripts.git
cd sigb_scripts
conda develop .
```
Since one of the goals is to keep setup in the working folders to a minimum, the following is recommended. In your working folder, create `job.sh` and `calc.py`. `job.sh` should manage the environments, and `calc.py` defines the specifics of this calculation, in addition to the generics provided in the pipeline script. 
```bash
# example job.sh on hoffman

#!/bin/bash -f
#$ -cwd
#$ -o $JOB_ID.log
#$ -e $JOB_ID.err
#$ -pe shared 32
#$ -l h_data=4G,h_rt=24:00:00

source /u/local/Modules/default/init/modules.sh 
source ~/.jobrc 
module purge
module load intel/2020.4 intel/mpi
conda activate ale

export OMP_NUM_THREAD=1
export I_MPI_COMPATIBILITY=4
export VASP_BIN='/u/home/x/xyttyxy/selfcompiled-programs/compile/vasp-5.4.4/source/bin/vasp_std_i2020.4_impi_vsol'
export VASP_COMMAND='mpirun -np ${NSLOTS} ${VASP_BIN}'
python calc.py
```
this calculation script is kept to the bare minimum because all the settings that should remain consistent across the whole project are moved to `pipelines.py`
```python3
from pipelines import freq
from ase.io import read

atoms = read('afterneb.vasp')
freq(atoms)
```
## Contributing
Changes to the source-controlled part (anything in this repo) should not be made to the master branch directly. For instance, when running jobs you discovered you need to change part of `pipelines.py`, you should create a separate branch via `git checkout -b <feature_branch>`, develop  your changes, and then commit the changes to `<feature_branch>`. Only when your are certain the changes *will not break existing `calc.py` scripts*, should you do:
```bash
git switch master
git merge <feature_branch>
git branch -D <feature_branch>
git push -u origin master
```

When you are not so certain changes will not be breaking, you can also push feature branch to remote:
```bash
# while on feature_branch
git push origin
```

by default this will create a remote branch called <feature_branch> and the local branch will be set to track this branch. This way other people will get the new code only when the specifically pull from this branch, thus not breaking their existing workflow. 

I (yantao) recommend creating a feature branch for each computer you use, and use that as the default branch on each computer. For instance when you use hoffman, bridges, and your own laptop, you can create 3 branches: hoff, br, laptop, all branching from master. Edits on each branch will not affect master. When they are ready, merge them to master and immediately switch back to the respective branch. 
