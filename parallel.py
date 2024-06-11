import javabridge
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

print(f"Script name: {sys.argv[0]}")
print(f"Input file: {sys.argv[1]}")

# Start the Java Virtual Machine
javabridge.start_vm(class_path=bioformats.JARS)
print("Java VM started successfully")

def get_image_dimensions(filepath):
    xml = bioformats.get_omexml_metadata(filepath)
    d = xml.split(' SizeX="')[1].split('"')[:3]
    return int(d[0]), int(d[2])

try:
    print(f"Processing file: {filepath}")
    size_x, size_y = get_image_dimensions(filepath)
    img = np.zeros((size_y, size_x, 3), dtype=np.uint8)
    with bioformats.ImageReader(filepath) as reader:
        tile_size = 4096
        for i in range(0, size_x, tile_size):
            for j in range(0, size_y, tile_size):
                tile = reader.read(XYWH=(i, j, min(tile_size, size_x - i), min(tile_size, size_y - j)))
                tile = (tile * 255).astype(np.uint8)
                img[j:j + tile.shape[0], i:i + tile.shape[1], :] = tile
    output_filename = os.path.basename(filepath).replace('.dcm', '.tif')
    output_path = os.path.join(output_dir, output_filename)
    tifffile.imsave(output_path, img)
    print(f"Saved TIFF at {output_path}")
except Exception as e:
    print(f"An error occurred while processing {filepath}: {e}")

# Stop the Java Virtual Machine
javabridge.kill_vm()
