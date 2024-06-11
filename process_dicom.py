import os
import sys
import pydicom
from pathlib import Path

def process_dicom_files(s3_url, base_dir):
    # Extract a safe directory name from the S3 URL
    dir_name = s3_url.split('/')[-2]  # Use the penultimate part of the URL as a directory name
    temp_dir = os.path.join(base_dir, dir_name)

    # Create a subdirectory for this URL's DICOM files
    os.makedirs(temp_dir, exist_ok=True)

    # Download all DICOM files from the S3 URL to the temporary directory
    # This part assumes you have access to the `s3_url` variable and s5cmd is installed
    os.system(f"s5cmd --no-sign-request --endpoint-url https://s3.amazonaws.com cp '{s3_url}' '{temp_dir}'")

    largest_file = None
    largest_size = 0

    # Process all downloaded DICOM files
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        try:
            # Attempt to read the DICOM file
            ds = pydicom.dcmread(file_path)

            # Keep track of the largest file
            file_size = os.path.getsize(file_path)
            if file_size > largest_size:
                largest_file = file_path
                largest_size = file_size
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Rename the largest file to its PatientID
    if largest_file:
        ds = pydicom.dcmread(largest_file)
        patient_id = ds.PatientID
        new_file_name = f"{patient_id}.dcm"
        new_file_path = os.path.join(temp_dir, new_file_name)
        os.rename(largest_file, new_file_path)
        print(f"Renamed largest file to {new_file_name} in {temp_dir}")

if __name__ == "__main__":
    s3_url = sys.argv[1]
    base_dir = sys.argv[2]

    process_dicom_files(s3_url, base_dir)
