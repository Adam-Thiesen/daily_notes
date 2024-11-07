import javabridge
import xml.etree.ElementTree as ET
import bioformats
import sys
import os

print(f"Script name: {sys.argv[0]}")
print(f"Input file: {sys.argv[1]}")

# Get file path and output directory from command line arguments
filepath = sys.argv[1]
output_dir = sys.argv[2]

base_filename = os.path.splitext(os.path.basename(filepath))[0]
#xml_filename = os.path.join(output_dir, f"{base_filename}_metadata.xml")
xml_filename = f'/flashscratch/thiesa/Containers/XML_metadata/{base_filename}.xml'

# Start the Java Virtual Machine
javabridge.start_vm(class_path=bioformats.JARS)
print("Java VM started successfully")

def save_metadata_if_contains_keywords(filepath, xml_filename):
    try:
        xml = bioformats.get_omexml_metadata(filepath)
        print("XML metadata obtained")

        # Check for specified keywords in XML metadata (case-insensitive)
        if ("sarcoma" in xml.lower() or "rhabdomyosarcoma" in xml.lower()) and "histiocytic" not in xml.lower():
            # Save XML metadata if it meets the criteria
            with open(xml_filename, 'w') as f:
                f.write(xml)
            print(f"XML metadata written to file: {xml_filename}")
        else:
            print("Metadata does not meet keyword criteria; not saved.")

    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")

try:
    print(f"Processing file: {filepath}")
    save_metadata_if_contains_keywords(filepath, xml_filename)
except Exception as e:
    print(f"An error occurred while processing {filepath}: {e}")

# Stop the Java Virtual Machine
javabridge.kill_vm()
