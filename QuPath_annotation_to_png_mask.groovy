import qupath.lib.gui.scripting.QPEx
import qupath.lib.roi.RoiTools
import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// --- Configuration ---
// Map QuPath classification names to specific pixel values (IDs)
def CLASS_MAP = [
    "blanc": 1,
] as Map

// Set image scaling: 1.0 is full resolution; >1 reduces size to save memory
double downsample = 1.0 

// Absolute destination path for the generated masks
def outDirAbsolute = "C:/Users/Bicel service/Desktop/USER/Maria/dataset"

// --- Image Verification ---
def imageData = getCurrentImageData()
if (imageData == null) {
    print 'Error: No image open in QuPath'
    return
}

// Extract server metadata and original file name
def server = imageData.getServer()
def metadata = server.getMetadata()
def imageNameFull = metadata.getName()

// --- Filename Sanitization ---
// Remove extensions and illegal characters to create a clean base name
def baseName = imageNameFull
baseName = baseName.replaceAll(/(?i)\.nd2$/, '')
baseName = baseName.replaceAll(/(?i)\.tif$/, '')
baseName = baseName.replaceAll(/(?i)\.tiff$/, '')
baseName = baseName.replaceAll(/(?i)\.png$/, '')

// Split name at " - " if present and remove extra spaces/special characters
if (baseName.contains(' - '))
    baseName = baseName.split(' - ')[0]
baseName = baseName.replaceAll(/\s.*$/, '')
baseName = baseName.replaceAll(/\s+/, '_')
baseName = baseName.replaceAll(/[^A-Za-z0-9_\-\.]/, '')
baseName = baseName.replaceAll(/_+/, '_')

// --- Mask Initialization ---
// Calculate dimensions based on the chosen downsample factor
int fullW = (int) Math.round(server.getWidth() / downsample)
int fullH = (int) Math.round(server.getHeight() / downsample)

// Create an 8-bit grayscale BufferedImage
def mask = new BufferedImage(fullW, fullH, BufferedImage.TYPE_BYTE_GRAY)
def gMask = mask.createGraphics()

// Set background to black (value 0)
gMask.setColor(Color.BLACK)
gMask.fillRect(0, 0, fullW, fullH)

// Apply scaling for the drawing process
gMask.scale(1.0 / downsample, 1.0 / downsample)

// --- Annotation Processing ---
// Retrieve all annotation objects from the image hierarchy
def annotations = imageData.getHierarchy().getAnnotationObjects()
def classesTrouvees = [] as Set // Keep track of classes found for folder creation

annotations.each { pathObj ->
    // Get the Region of Interest (ROI)
    def roi = pathObj.getROI()
    if (roi == null) return
    
    // Check if the annotation class exists in our CLASS_MAP
    def pathClass = pathObj.getPathClass()
    def clsName = pathClass == null ? 'None' : pathClass.toString()
    
    if (!CLASS_MAP.containsKey(clsName)) {
        print "Class '${clsName}' not mapped -> ignored"
        return
    }
    
    classesTrouvees << clsName
    
    // Map the class ID to a grayscale pixel value (v,v,v)
    int clsId = CLASS_MAP[clsName]
    int v = Math.max(1, Math.min(254, clsId))
    def color = new Color(v, v, v)
    
    // Convert the QuPath ROI into a Java AWT Shape for rendering
    def shape = RoiTools.getShape(roi)
    if (shape == null) return

    // Draw and fill the annotation on the mask with the assigned ID value
    gMask.setColor(color)
    gMask.fill(shape)
}

// Finalize the graphics context
gMask.dispose()

// --- Export ---
// Save the generated mask into specific sub-folders named after each class
classesTrouvees.each { clsName ->
    def classDir = new File(outDirAbsolute, "masks_" + clsName)
    if (!classDir.exists()) {
        classDir.mkdirs()
    }
    
    // Export as PNG (Lossless) using the cleaned base name
    def outFile = new File(classDir, baseName + ".png")
    ImageIO.write(mask, "PNG", outFile)
    print "Mask successfully saved to: " + outFile.getAbsolutePath()
}