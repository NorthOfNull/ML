#!/bin/sh

#SBATCH --job-name=E-CLF-CV
#SBATCH --output CLF-CV-EXT-%j.txt
#SBATCH --time=5-00:00:00

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6GB

#SBATCH --partition=120hour
#SBATCH --mail-user kyle.fryer@northumbria.ac.uk
#SBATCH --mail-type=ALL

module load anaconda3
source activate ml
cd ML

mpirun -n 15 python3 hpc_clf_cv_ext.py 
