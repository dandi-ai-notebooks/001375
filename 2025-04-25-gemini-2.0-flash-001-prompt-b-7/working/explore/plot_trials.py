import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to load NWB file and plot trials data (start_time vs. stop_time)

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access trials data
trials = nwb.trials
start_time = trials.start_time[:]
stop_time = trials.stop_time[:]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(start_time, stop_time)
plt.xlabel("Start Time (s)")
plt.ylabel("Stop Time (s)")
plt.title("Trials: Start Time vs. Stop Time")
plt.savefig("explore/trials_start_stop.png")
plt.close()