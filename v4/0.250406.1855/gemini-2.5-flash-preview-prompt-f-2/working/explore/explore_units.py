# This script explores the units data (spike times).

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the units data
units_df = nwb.units.to_dataframe()
print("Units DataFrame Head:")
print(units_df.head())

# Calculate spike counts per unit
spike_counts = units_df['spike_times'].apply(len)

# Plot histogram of spike counts
plt.figure(figsize=(10, 6))
plt.hist(spike_counts, bins=20)
plt.xlabel("Number of Spikes")
plt.ylabel("Number of Units")
plt.title("Histogram of Spike Counts per Unit")
plt.grid(True)
plt.savefig('explore/spike_counts_histogram.png')

io.close()