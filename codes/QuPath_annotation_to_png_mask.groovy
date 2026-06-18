import qupath.lib.gui.scripting.QPEx
import qupath.lib.roi.RoiTools
import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// --- Configuration à adapter ---
def CLASS_MAP = [
    "blanc": 1,
    "alteration": 2,
    "canaux_normaux": 3,
] as Map

double downsample = 1.0 // 1.0 = résolution native ; >1 = downsample

// Chemin adapté à ton PC actuel
def outDirAbsolute = "C:/Users/Bicel service/Desktop/USER/Maria/dataset"

// -------------------------------
def imageData = getCurrentImageData()
if (imageData == null) {
    print 'Erreur : aucune image ouverte dans QuPath'
    return
}

def server = imageData.getServer()
def metadata = server.getMetadata()
def imageNameFull = metadata.getName()

// Nettoyage du nom de l’image
def baseName = imageNameFull
baseName = baseName.replaceAll(/(?i)\.nd2$/, '')
baseName = baseName.replaceAll(/(?i)\.tif$/, '')
baseName = baseName.replaceAll(/(?i)\.tiff$/, '')
baseName = baseName.replaceAll(/(?i)\.png$/, '')
if (baseName.contains(' - '))
    baseName = baseName.split(' - ')[0]
baseName = baseName.replaceAll(/\s.*$/, '')
baseName = baseName.replaceAll(/\s+/, '_')
baseName = baseName.replaceAll(/[^A-Za-z0-9_\-\.]/, '')
baseName = baseName.replaceAll(/_+/, '_')

// Taille du masque
int fullW = (int) Math.round(server.getWidth() / downsample)
int fullH = (int) Math.round(server.getHeight() / downsample)

// Créer une seule image masque (8-bit)
def mask = new BufferedImage(fullW, fullH, BufferedImage.TYPE_BYTE_GRAY)
def gMask = mask.createGraphics()
gMask.setColor(Color.BLACK)
gMask.fillRect(0, 0, fullW, fullH)
gMask.scale(1.0 / downsample, 1.0 / downsample)

// Parcourir toutes les annotations et les peindre
def annotations = imageData.getHierarchy().getAnnotationObjects()
def classesTrouvees = [] as Set

annotations.each { pathObj ->
    def roi = pathObj.getROI()
    if (roi == null) return
    def pathClass = pathObj.getPathClass()
    def clsName = pathClass == null ? 'None' : pathClass.toString()
    if (!CLASS_MAP.containsKey(clsName)) {
        print "Classe non mappée pour '${clsName}' -> ignorée"
        return
    }
    classesTrouvees << clsName
    int clsId = CLASS_MAP[clsName]
    int v = Math.max(1, Math.min(254, clsId))
    def color = new Color(v, v, v)
    def shape = RoiTools.getShape(roi)
    if (shape == null) return

    gMask.setColor(color)
    gMask.fill(shape)
}

gMask.dispose()

// Sauvegarder le masque dans chaque dossier de classe annotée
classesTrouvees.each { clsName ->
    def classDir = new File(outDirAbsolute, "masks_" + clsName)
    if (!classDir.exists()) {
        classDir.mkdirs()
    }
    def outFile = new File(classDir, baseName + ".png")  // même nom que l’image
    ImageIO.write(mask, "PNG", outFile)
    print "Masque sauvegardé : " + outFile.getAbsolutePath()
}
