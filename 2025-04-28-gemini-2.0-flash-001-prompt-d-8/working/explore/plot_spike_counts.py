# explore/plot_spike_counts.py
# This script loads the NWB file and plots the number of spikes for the first 20 units in the 5-10 second interval

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
unit_ids = units.id

# Time window
start_time = 5  # seconds
end_time = 10  # seconds

# Calculate spike counts for the first 20 units
num_units_to_plot = 20
spike_counts = []
for i in range(num_units_to_plot):
    unit_spike_times = spike_times[i]
    spike_times_in_window = unit_spike_times[(unit_spike_times >= start_time) & (unit_spike_times <= end_time)]
    spike_counts.append(len(spike_times_in_window))

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(range(num_units_to_plot), spike_counts, tick_label=unit_ids[:num_units_to_plot])
plt.xlabel("Unit ID")
plt.ylabel("Number of Spikes in 5-10s")
plt.title("Spike Counts for First 20 Units (5-10s)")
plt.savefig("explore/spike_counts_plot.png")
plt.close()