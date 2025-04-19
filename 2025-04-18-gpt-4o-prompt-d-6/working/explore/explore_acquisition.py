# This script explores the reduced acquisition data within the NWB file.
# The aim is to extract and visualize a manageable subset of multi-channel information in the acquisition time series data for initial analysis.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Explore a reduced subset of the acquisition data
subset_data = nwb.acquisition["time_series"].data[0:6000, 0:10]  # First 200 ms with 10 channels
times = np.arange(subset_data.shape[0]) / nwb.acquisition["time_series"].rate

# Plot the first few channels from the subset data
plt.figure(figsize=(12, 6))
for i in range(subset_data.shape[1]):  # Plotting the first 10 channels
    plt.plot(times, subset_data[:, i] + i * 10, label=f"Channel {i}")  # Shift each channel for readability

plt.title("Subset Multi-channel Time Series Overview")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.legend(loc="upper right", fontsize="small")
plt.savefig("explore/acquisition_subset_overview.png")
plt.close()

# Closing the file connections
io.close()
h5_file.close()