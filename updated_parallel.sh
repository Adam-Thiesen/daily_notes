#!/bin/bash

# Set the base directory and output directory
BASE_DIR="/flashscratch/thiesa/download_images2/all_dicom_files2"
OUTPUT_DIR="${BASE_DIR}/tif_images"

# Ensure the output directory exists
mkdir -p ${OUTPUT_DIR}

# Pre-compute the number of RMS files for the array job
NUM_FILES=$(find ${BASE_DIR} -type f -name '*RMS*.dcm' | wc -l)
NUM_FILES=$(($NUM_FILES - 1)) # Adjust for zero indexing

# Dynamically set the SBATCH options and submit the job
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=image_processing
#SBATCH --time=3:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --array=0-$NUM_FILES
#SBATCH -p compute
#SBATCH --output=slurm-%A_%a.out

# Find all RMS files
FILES=(\$(find $BASE_DIR -type f -name '*RMS*.dcm'))

# Determine the file to be processed based on the array index
FILE_TO_PROCESS=\${FILES[\$SLURM_ARRAY_TASK_ID]}

echo "File to process: \$FILE_TO_PROCESS"
echo "Output directory: $OUTPUT_DIR"

# Load necessary modules
module load singularity

# Execute the Python script with the specific file to process
singularity exec /flashscratch/thiesa/download_images2/updated_bioformats.sif python /flashscratch/thiesa/download_images2/parallel.py "\$FILE_TO_PROCESS" "$OUTPUT_DIR"
EOT
