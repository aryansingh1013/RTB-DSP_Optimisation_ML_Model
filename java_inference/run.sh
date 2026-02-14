#!/bin/bash
# Run script for RTB DSP Engine
# -Xms512m -Xmx512m: Lock heap memory to 512MB
# -XX:+UseSerialGC: Use low-overhead Serial Garbage Collector for responsiveness
# -cp submission.jar: Add our jar to classpath

# Ensure weights are present in the current directory (copied from python_training usually)
if [ ! -f ctr_weights.csv ]; then
    echo "WARNING: ctr_weights.csv not found! Please run the Python notebook and copy weights here."
fi

java -Xms512m -Xmx512m -XX:+UseSerialGC -cp submission.jar com.dtu.hackathon.bidding.Bid
