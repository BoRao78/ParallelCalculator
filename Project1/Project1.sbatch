#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --nodes=1
#SBATCH --chdir=/home/oucs023/Project1
#SBATCH --output=/home/oucs023/Project1/stdout_Project1_%J.txt
#SBATCH --error=/home/oucs023/Project1/stderr_Project1_%J.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=Project1
#SBATCH --mail-user=Bo.Rao-1@ou.edu
#SBATCH --mail-type=ALL
#
##################################################################

echo "Working directory:"
pwd
echo ""

# Set the runtime environment to match the compile environment.

module load intel

echo "Project1 Algorithm 2, Matrix size 100x100 to 1000x10000"

mpirun -np 1 /home/oucs023/Project1/Project1A2
mpirun -np 4 /home/oucs023/Project1/Project1A2
mpirun -np 9 /home/oucs023/Project1/Project1A2
mpirun -np 16 /home/oucs023/Project1/Project1A2
mpirun -np 25 /home/oucs023/Project1/Project1A2
mpirun -np 36 /home/oucs023/Project1/Project1A2
mpirun -np 64 /home/oucs023/Project1/Project1A2

echo "Project1 Fox's Algorithm, Matrix size 100x100 to 1000x10000"

mpirun -np 1 /home/oucs023/Project1/Project1Fox
mpirun -np 4 /home/oucs023/Project1/Project1Fox
mpirun -np 9 /home/oucs023/Project1/Project1Fox
mpirun -np 16 /home/oucs023/Project1/Project1Fox
mpirun -np 25 /home/oucs023/Project1/Project1Fox
mpirun -np 36 /home/oucs023/Project1/Project1Fox
mpirun -np 64 /home/oucs023/Project1/Project1Fox
