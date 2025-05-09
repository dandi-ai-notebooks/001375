# Explore unit spike times in NWB file
# Output: print summary of units table, save histogram of spike times for first five units
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

units_df = nwb.units.to_dataframe()
print(f"Units table shape: {units_df.shape}")
print("First 5 rows of units table (showing spike counts):")
for i in range(min(5, len(units_df))):
    spike_times = units_df.iloc[i]['spike_times']
    if isinstance(spike_times, np.ndarray):
        n_spikes = len(spike_times)
    else:
        n_spikes = "N/A"
    print(f"Unit {units_df.index[i]}: {n_spikes} spikes")

# Plot histogram of spike times for first 5 units over first 100 sec
plt.figure(figsize=(10,6))
for i in range(min(5, len(units_df))):
    spike_times = units_df.iloc[i]['spike_times']
    if isinstance(spike_times, np.ndarray):
        valid = spike_times[spike_times < 100]
        plt.hist(valid, bins=100, alpha=0.5, label=f'Unit {units_df.index[i]}')
plt.xlabel("Time (s)")
plt.ylabel("Spike count")
plt.title("Spike time histogram, first 100s, first 5 units")
plt.legend()
plt.tight_layout()
plt.savefig("explore/spike_histogram.png")
print("Saved plot to explore/spike_histogram.png")

io.close()
h5_file.close()
remote_file.close()