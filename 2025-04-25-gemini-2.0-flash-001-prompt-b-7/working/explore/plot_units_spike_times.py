import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to load NWB file and plot a histogram of units spike_times

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access units data
units = nwb.units
spike_times = units.spike_times[:]

# Create histogram of spike times
plt.figure(figsize=(10, 6))
plt.hist(spike_times, bins=50)
plt.xlabel("Spike Times (s)")
plt.ylabel("Number of Spikes")
plt.title("Distribution of Spike Times across Units")
plt.savefig("explore/units_spike_times_histogram.png")
plt.close()