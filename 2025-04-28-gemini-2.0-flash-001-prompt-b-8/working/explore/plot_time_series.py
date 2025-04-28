import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# This script loads a small segment of the time_series data from the NWB file and plots it.
# The plot is saved as a PNG file in the explore/ directory.

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Load a small segment of the time_series data (first 1000 samples from the first channel).
data = nwb.acquisition["time_series"].data[:1000, 0]
timestamps = np.linspace(0, len(data) / nwb.acquisition["time_series"].rate, len(data))

# Plot the data.
plt.figure(figsize=(10, 5))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Time Series Data (First 1000 Samples, Channel 0)")
plt.savefig("explore/time_series.png")
plt.close()