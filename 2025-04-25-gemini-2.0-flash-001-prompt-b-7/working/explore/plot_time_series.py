import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Plotting with seaborn styling conflicts image masks values
# sns.set_theme()

# Script to load NWB file and plot a segment of time_series.data

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access time_series data
time_series = nwb.acquisition["time_series"]
data = time_series.data
rate = time_series.rate

# Select a segment of data (e.g., first 10 seconds, first channel)
duration = 10  # seconds
start_time = 0  # seconds
start_index = int(start_time * rate)
end_index = int((start_time + duration) * rate)
channel_index = 0
segment = data[start_index:end_index, channel_index]
time = np.arange(start_time, start_time + duration, 1/rate)

# Plot the segment
plt.figure(figsize=(10, 5))
plt.plot(time, segment)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Segment of time_series.data")
plt.savefig("explore/time_series_segment.png")
plt.close()