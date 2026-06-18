*If you want to plot the histogram of a BrightField sample using ImageJ/Fiji :* 



open("C:/Users/Bicel service/Desktop/USER/Maria/data/25-11-2025/koek-4/koek-4-stitch-bf.nd2");

selectImage("koek-4-stitch-bf.nd2");

setAutoThreshold("Triangle dark no-reset");

//run("Threshold...");

run("Convert to Mask", "method=Triangle background=Dark calculate black create");

run("Invert", "slice");

run("Fill Holes", "slice");

run("Analyze Particles...", "size=1000000-Infinity show=Masks clear include add slice");

selectImage("Mask of MASK\_koek-4-stitch-bf.nd2");

imageCalculator("AND create 32-bit", "koek-4-stitch-bf.nd2","Mask of MASK\_koek-4-stitch-bf.nd2");

selectImage("Result of koek-4-stitch-bf.nd2");

setAutoThreshold("Triangle dark no-reset");

selectImage("Result of anat-2-stitch-bf.nd2");

run("Histogram", "bins=256 use x\_min=3 x\_max=237 y\_max=Auto");



