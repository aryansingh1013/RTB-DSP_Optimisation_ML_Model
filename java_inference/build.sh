#!/bin/bash
# Clean previous build
rm -rf bin
rm -f submission.jar

# Create output directory
mkdir -p bin

# Compile Java Code
# -d bin: output class files to bin directory
javac -d bin src/com/dtu/hackathon/bidding/Bid.java

# Copy weight files to the root of the class path (inside the JAR conceptually or just in the folder)
# In this setup, we assume weight files are in the same folder as the jar or explicitly loaded.
# Bid.java loads from filenames directly, so they should be in the working directory when running java.
# We don't strictly need to put them in the JAR for the code to run if we unzip it, 
# but for submission (single file), usually resources are embedded. 
# My Bid.java uses 'new File("ctr_weights.csv")', so files must be on disk.
# We will just package the classes.

jar cvf submission.jar -C bin .

echo "Build complete. Created submission.jar"
