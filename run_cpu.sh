#!/bin/bash --login

#!/bin/bash --login
#SBATCH --job-name="jjup"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH -e err_jupyter.err
#SBATCH -o out_sumner_jupyter.out

# Load the CUDA module
# module load cuda12.1/toolkit/12.1.1

cd $SLURM_SUBMIT_DIR
unset XDG_RUNTIME_DIR

IP=`hostname -i`
PORT=$(shuf -i10000-11999 -n1)

bindDir="/sdata/"
notebookDir="/flashscratch/thiesa/Pytorch3"
container="/flashscratch/thiesa/Pytorch3/pytorch.sif"

#singularity exec --nv -B "${bindDir}" "${container}" jupyter notebook --no-browser --port=$PORT --ip=$IP --notebook-dir "${notebookDir}"
singularity exec -B "${bindDir}" "${container}" jupyter lab --no-browser --port=$PORT --ip=$IP --notebook-dir "${notebookDir}"
