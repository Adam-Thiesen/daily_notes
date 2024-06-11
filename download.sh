#!/bin/bash
#SBATCH --job-name=download_images
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH -p compute

# Path to the Singularity container
SINGULARITY_CONTAINER="enviro.sif"

# Base directory to hold all subdirectories with DICOM files
BASE_DIR="all_dicom_files13"
mkdir -p $BASE_DIR


# Array of S3 URLs
S3_URLS=(
    s3://idc-open-data/de6a5021-3318-4457-bc58-46216e14ba92/*
    s3://idc-open-data/7df7ebab-994f-4b05-92ef-d12eb73be132/*
    s3://idc-open-data/76ca2417-c53d-4b20-9cdb-b2b3967b9134/*
    s3://idc-open-data/96f3bb22-63e4-4df9-ba22-974f64edcee6/*
    s3://idc-open-data/3df5fc61-3b81-4b2f-8b5f-6895d5494ae0/*
    s3://idc-open-data/e3a6d2f7-7586-4b0b-b214-78766cfa5a11/*
    s3://idc-open-data/9365e6a1-8100-4163-9a79-a3fdfffe0449/*
    s3://idc-open-data/4aa93a00-34bd-4db4-8490-26d288e9a745/*
    s3://idc-open-data/404f17cf-cea4-4f77-b800-8784fce88821/*
    s3://idc-open-data/48bd43ac-aa7f-4365-9a18-8eb0d12ea556/*
    s3://idc-open-data/6290f954-a2e7-4433-a9f5-93a7d68a7859/*
    s3://idc-open-data/d499c95a-3505-49c6-873c-957713939e4f/*
    s3://idc-open-data/d7c86151-0558-4caf-84da-8b954442d37a/*
    s3://idc-open-data/0e9c4e62-4ca6-4000-9d45-f7c3ffec50a1/*
    s3://idc-open-data/20c44a5d-4607-4d79-9ec4-0155a9608d6d/*
    s3://idc-open-data/e104aa89-c890-4366-9b30-c8f9f898b4ae/*
    s3://idc-open-data/c072c7ce-43e5-43a3-97f7-c1e6ccc074c2/*
    s3://idc-open-data/9c12d291-4aa1-42d6-a533-ddcb74eecd46/*
    s3://idc-open-data/c2cba930-d1a7-42f8-92ef-98be2126dfea/*
    s3://idc-open-data/ccbb151a-11f6-4d99-bbb6-390a1b481d2f/*
    s3://idc-open-data/e7927577-d906-4804-80f0-a1e166101874/*
    s3://idc-open-data/a73e397f-6c51-4dc8-8a79-d0f4f0e2a5d9/*
    s3://idc-open-data/f15e0fac-1eff-4ceb-b89e-c3bfe949a94f/*
    s3://idc-open-data/a83f4398-0891-4a85-a8aa-4e6a1e8bd28d/*
    s3://idc-open-data/a42f8661-06e0-44de-b18d-717bf7be1ada/*
    s3://idc-open-data/a3308e07-1d3c-429a-89f2-b4515cbff64c/*
    s3://idc-open-data/c9698fa6-9438-47cd-a835-7e581d366b1e/*
    s3://idc-open-data/4576c007-8fee-4df6-896c-2cbdfb95f677/*
)

# Loop through each S3 URL
for S3_URL in "${S3_URLS[@]}"
do
    echo "Processing $S3_URL..."

    # Run the Singularity container with the S3 URL and base directory
    # The Python script inside the container will handle the creation of subdirectories, downloading, and file processing
    singularity exec $SINGULARITY_CONTAINER python /flashscratch/thiesa/download_images/process_dicom.py "$S3_URL" $BASE_DIR
done

echo "Processing complete."
