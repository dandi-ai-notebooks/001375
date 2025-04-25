import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract a small segment of the time_series.data
time_series = nwb.acquisition["time_series"]
data = time_series.data[:1000, :5]  # First 1000 time points and first 5 channels
rate = time_series.rate

# Generate time axis
time = np.arange(0, len(data)) / rate

# Plot the data
plt.figure(figsize=(10, 6))
for i in range(data.shape[1]):
    plt.plot(time, data[:, i] + i * 100, label=f"Channel {i}") # adding offset for each channel
plt.xlabel("Time (s)")
plt.ylabel("mV + offset")
plt.title("Time series data for the first 5 channels")
plt.legend()
plt.savefig("timeseries.png")
plt.close()