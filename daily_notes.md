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

2/27/24
--------------------------------
The dicom files from the paper on rhabo mutational prediction all seem to be exactly the same, so it will be difficult to tell which ones are the real images. Additionally, these are multiframe dicom images that are tiled into 1080 240x240 tiles. I can convert them to Tiff and get it to somewhat work, but the image size is quite small.

It may be easiest to just do everything manually in a day or 2. I was able to read in the file completely using QuPath version 0.5.0, then export to OME-TIFF. So, everything seems to be fine doing it that way. One question is that QuPath is showing variable pixel width and pixel height, so I may need to check and make sure that we have the right mpp, for me it is showing 1.0105, but Sergii was showing 0.25.


3/19/24
----------------------------------
When I use bfconvert to convert the .dcm to .tiff there is no parameter to save the .dcm as a pyramid. When I read in the file using pillow, it has bad resolution, looks green, and important metadata is not saved. 

When I open the largest image file on qupath v0.5 it opens perfectly. When I export as ome.tiff with 4x downsampling, it also opens perfectly. Therefore, as a worst case scenario, doing everything through qupath may be an option.

Next, I will need to learn about qupath scripting. 
