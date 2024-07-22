import javabridge
import xml.etree.ElementTree as ET
import bioformats
import numpy as np
import tifffile
import sys
import os

print(f"Script name: {sys.argv[0]}")
print(f"Input file: {sys.argv[1]}")

# Get file path and output directory from command line arguments
filepath = sys.argv[1]
output_dir = sys.argv[2]

base_filename = os.path.splitext(os.path.basename(filepath))[0]
xml_filename = f'/flashscratch/thiesa/download_images2/XML_metadata/{base_filename}_metadata.xml'


print(f"Script name: {sys.argv[0]}")
print(f"Input file: {sys.argv[1]}")

# Start the Java Virtual Machine
javabridge.start_vm(class_path=bioformats.JARS)
print("Java VM started successfully")

def get_image_dimensions(filepath, xml_filename):
    try:
        xml = bioformats.get_omexml_metadata(filepath)
        print("XML metadata obtained")

        # Write XML to file for debugging or inspection
        with open(xml_filename, 'w') as f:
            f.write(xml)
        print("XML metadata written to file")

        # Parse the XML using ElementTree
        root = ET.fromstring(xml)
        # Navigate to the Pixels element where dimensional data is stored
        pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
        
        # Extract dimensions
        size_x = pixels.get('SizeX')
        size_y = pixels.get('SizeY')

        return int(size_x), int(size_y)
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")
        return None, None

try:
    print(f"Processing file: {filepath}")
    size_x, size_y = get_image_dimensions(filepath, xml_filename)
    img = np.zeros((size_y, size_x, 3), dtype=np.uint8)
    with bioformats.ImageReader(filepath) as reader:
        tile_size = 4096
        for i in range(0, size_x, tile_size):
            for j in range(0, size_y, tile_size):
                try:
                    tile = reader.read(XYWH=(i, j, min(tile_size, size_x - i), min(tile_size, size_y - j)))
                    tile = (tile * 255).astype(np.uint8)
                    img[j:j + tile.shape[0], i:i + tile.shape[1], :] = tile
                except:
                    print(f"An error occurred while processing {filepath}: {e} at {i}, {j}")
    output_filename = os.path.basename(filepath).replace('.dcm', '.tif')
    output_path = os.path.join(output_dir, output_filename)
    tifffile.imsave(output_path, img)
    print(f"Saved TIFF at {output_path}")
except Exception as e:
    print(f"An error occurred while processing {filepath}: {e}")

# Stop the Java Virtual Machine
javabridge.kill_vm()
