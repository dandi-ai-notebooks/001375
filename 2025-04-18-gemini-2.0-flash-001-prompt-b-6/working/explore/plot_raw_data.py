# %%
# This script loads raw data from an NWB file and plots it.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get raw data
raw_data = nwb.acquisition["time_series"].data[:1000, 0]
timestamps = np.linspace(0, len(raw_data)/nwb.acquisition["time_series"].rate, len(raw_data))

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(timestamps, raw_data)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Raw Data from Channel 0")
plt.savefig("explore/raw_data.png")
plt.close()