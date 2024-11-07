#!/bin/bash

# Set the base directory and output directory
BASE_DIR="/flashscratch/thiesa/Containers/dicom_files_new2"
OUTPUT_DIR="${BASE_DIR}/tif_images_converted"

# Ensure the output directory exists
mkdir -p ${OUTPUT_DIR}

# Pre-compute the number of files with names less than 10 characters for the array job
NUM_FILES=$(find ${BASE_DIR} -type f -name '??????.dcm' | wc -l)
echo "Number of files: $NUM_FILES"
NUM_FILES=$(($NUM_FILES - 1)) # Adjust for zero indexing

# Dynamically set the SBATCH options and submit the job
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=image_processing
#SBATCH --time=3:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --array=0-$NUM_FILES%10
#SBATCH -p compute
#SBATCH --output=slurm-%A_%a.out

# Find all files with names less than 10 characters (excluding the extension)
FILES=(\$(find $BASE_DIR -type f -name '??????.dcm'))


# Determine the file to be processed based on the array index
FILE_TO_PROCESS=\${FILES[\$SLURM_ARRAY_TASK_ID]}

echo "File to process: \$FILE_TO_PROCESS"
echo "Output directory: $OUTPUT_DIR"

# Load necessary modules
module load singularity

# Execute the Python script with the specific file to process
singularity exec /flashscratch/thiesa/Containers/updated_bioformats.sif python /flashscratch/thiesa/Containers/get_xml_metadata.py "\$FILE_TO_PROCESS" "$OUTPUT_DIR"
EOT
