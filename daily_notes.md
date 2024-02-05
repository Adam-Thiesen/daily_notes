1/31/24
---------------------
Feature extraction script is taking over an hour to run on winter, will need to check if there is a way to reduce this time. The images for feature extraction are quite large (>1gb), so that might be contributing.

In .sh bash files, to get the error output and regular slurm output use this:
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err (with this you can print out errors to the output script)

For pancreas visium analysis, I will need to check if there is a way to get the spatial coordinates of the spots, then the spatial coordinates of each cell, and see if cells within specific spatial regions have varying morphology.

Hovernet3 is my environment that has openslide, timm, torch, and tensorflow, so pretty much everything

2/5/24
----------------------------
When running the STQ pipeline to just extract inception features, make sure to comment out 
withLabel: process_ctranspath {
        cpus = 10
        container = params.container_ctranspath        
        if (params.ctranspath_device_mode == 'gpu') {
            cpus = 16
            clusterOptions = '--time=02:00:00 -q gpu_inference --gres=gpu:1 --export=ALL'
            containerOptions = '--nv'
            queue = 'gpus'
        }
        else if (params.ctranspath_device_mode == 'cpu') {
            cpus = 10
            clusterOptions = '--time=16:00:00'
        }
    }
    in nextflow.config 
