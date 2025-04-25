# Explore spike times for selected units from Dandiset 001375, asset ce525828-8534-4b56-9e47-d2a34d1aa897
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get units data
units = nwb.units
num_units = len(units.id)
print(f"Found {num_units} units.")

# Select first 5 units to plot
num_units_to_plot = min(5, num_units)
unit_ids_to_plot = units.id[:num_units_to_plot]
print(f"Plotting spike times for the first {num_units_to_plot} units:")
print(unit_ids_to_plot)

# Define time range for plotting
plot_start_time = 0.0
plot_end_time = 100.0 # Plot first 100 seconds

# Create raster plot
sns.set_theme()
plt.figure(figsize=(12, 6))
for i, unit_id in enumerate(unit_ids_to_plot):
    unit_index = np.where(units.id[:] == unit_id)[0][0]
    spike_times = units['spike_times'][unit_index]
    # Filter spike times within the plot range
    spike_times_in_range = spike_times[(spike_times >= plot_start_time) & (spike_times < plot_end_time)]
    plt.plot(spike_times_in_range, np.ones_like(spike_times_in_range) * i, '|', markersize=5, label=f'Unit {unit_id}')

plt.yticks(range(num_units_to_plot), [f'Unit {uid}' for uid in unit_ids_to_plot])
plt.title(f'Spike Raster Plot ({plot_start_time}s - {plot_end_time}s)')
plt.xlabel('Time (s)')
plt.ylabel('Unit ID')
plt.ylim(-0.5, num_units_to_plot - 0.5)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('explore/spike_raster.png')
print("\nSaved plot to explore/spike_raster.png")

io.close()