input = getArgument();

// 1. Open
run("Bio-Formats Importer", "open=[" + input + "] autoscale color_mode=Default view=[Standard ImageJ] stack_order=XYCZT");
rename("original"); // Force a simple name

// 2. Create Mask
run("Duplicate...", "title=work");
setAutoThreshold("Triangle dark no-reset");
run("Convert to Mask", "method=Triangle background=Dark black");
run("Invert");
run("Fill Holes");

// 3. Analyze Particles
// We use 'show=[Masks]' which creates a window named 'Mask of work'
run("Analyze Particles...", "size=1000000-Infinity show=Masks clear include add");
selectWindow("Mask of work");

// 4. Combine
imageCalculator("AND create 32-bit", "original", "Mask of work");
selectWindow("Result of original");

run("Histogram", "bins=256 use x_min=3 x_max=237 y_max=Auto");


