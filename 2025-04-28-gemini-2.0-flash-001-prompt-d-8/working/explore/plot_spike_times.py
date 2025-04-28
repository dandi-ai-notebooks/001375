# explore/plot_spike_times.py
# This script loads the NWB file and plots a segment of the spike times data for a specific unit

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the units data
units = nwb.units
spike_times = units.spike_times

# Select a unit and a time window
unit_id = units.id[14]  # Selecting unit 15 (index 14)
start_time = 5  # seconds - selecting a short time window
end_time = 10  # seconds

# Get spike times for the selected unit within the time window
unit_spike_times = spike_times[14]
spike_times_in_window = unit_spike_times[(unit_spike_times >= start_time) & (unit_spike_times <= end_time)]


# Create the plot as a raster plot
plt.figure(figsize=(10, 4))
plt.eventplot(spike_times_in_window, linelengths=0.8, colors='k')  # Raster plot
plt.xlabel("Time (s)")
plt.ylabel("Unit")
plt.title(f"Spike Times for Unit {unit_id} ({start_time}-{end_time}s)")
plt.yticks([1], [unit_id])  # Show unit ID on y-axis
plt.savefig("explore/spike_times_plot.png")
plt.close()