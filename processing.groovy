import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.images.servers.bioformats.BioFormatsImageServer


// Define the main directory that contains subdirectories and the output directory
String mainDir = '/Users/thiesa/Desktop/all_dicom_files4'
String outputDir = '/Users/thiesa/Desktop/output_folder4'

// Iterate over each subdirectory in the main directory
new File(mainDir).eachDir { dir ->
    // Iterate over each file in the current subdirectory
    dir.eachFile { file ->
        // Check if the file starts with "RMS"
        if (file.name.startsWith("RMS")) {
            try (def server = new BioFormatsImageServer(file.toURI())) {
                // Define the output file path using the first 4 characters of the original file name
                String outputFileName = file.name.take(7) + '.ome.tiff'
                String outputPath = new File(outputDir, outputFileName).getAbsolutePath()
    
                OMEPyramidWriter.OMEPyramidSeries writer = new OMEPyramidWriter.Builder(server)
                	.scaledDownsampling(server.getDownsampleForResolution(0), 4)
                	.parallelize()
                	.build()

                OMEPyramidWriter.createWriter(writer).writeImage(outputPath)
            } catch (Exception e) {
                println("Error processing file: ${file.absolutePath}\n${e.message}")
            }
        }
    }
}
