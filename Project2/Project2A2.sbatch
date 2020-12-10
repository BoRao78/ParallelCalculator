#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --ntasks-per-node=20
#SBATCH --nodes=2 
#SBATCH --chdir=/home/oucs023/Project2
#SBATCH --output=/home/oucs023/Project2/stdout_Project2_%J.txt
#SBATCH --error=/home/oucs023/Project2/stderr_Project2_%J.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=Project2
#SBATCH --mail-user=Bo.Rao-1@ou.edu
#SBATCH --mail-type=ALL
#
##################################################################

echo "Working directory:"
pwd
echo ""



echo "Project1 Algorithm 2, Matrix size 100x100 to 1000x10000"

/home/oucs023/Project2/Project2A2 1
/home/oucs023/Project2/Project2A2 4
/home/oucs023/Project2/Project2A2 9
/home/oucs023/Project2/Project2A2 16
/home/oucs023/Project2/Project2A2 25
/home/oucs023/Project2/Project2A2 36
/home/oucs023/Project2/Project2A2 64
