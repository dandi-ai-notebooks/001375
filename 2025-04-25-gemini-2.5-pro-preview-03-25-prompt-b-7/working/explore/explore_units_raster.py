# Script to explore spike times from the NWB file
# https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/
# It generates a raster plot for the first 10 units over the first 60 seconds.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme for plotting
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
print("NWB file loaded.")

# Access the units table
units = nwb.units
print(f"Accessed 'units' table with columns: {units.colnames}")
print(f"Total number of units: {len(units.id)}")

# Define parameters
num_units_to_plot = 10
time_limit = 60.0  # seconds
unit_ids_to_plot = units.id[:num_units_to_plot] # Get the actual IDs
print(f"Plotting spike times for the first {num_units_to_plot} units (IDs: {list(unit_ids_to_plot)}) up to {time_limit} seconds.")

# Collect spike times for the selected units within the time limit
spike_times_list = []
unit_plot_indices = [] # Y-values for the raster plot
actual_unit_ids_plotted = [] # Keep track of which units actually have spikes

for i, unit_id in enumerate(unit_ids_to_plot):
    # Find the index corresponding to the unit_id
    unit_index = np.where(units.id[:] == unit_id)[0][0]
    
    # Get spike times for this unit index
    # Accessing spike times requires using the VectorIndex
    start_idx = units.spike_times_index.data[unit_index-1] if unit_index > 0 else 0
    end_idx = units.spike_times_index.data[unit_index]
    unit_spike_times = units.spike_times.data[start_idx:end_idx]

    # Filter spike times
    unit_spike_times_filtered = unit_spike_times[unit_spike_times <= time_limit]
    
    if len(unit_spike_times_filtered) > 0:
        spike_times_list.append(unit_spike_times_filtered)
        unit_plot_indices.append(i) # Use the loop index for plotting order
        actual_unit_ids_plotted.append(unit_id)
    print(f"Unit ID {unit_id}: Found {len(unit_spike_times_filtered)} spikes <= {time_limit}s.")


# Create and save the raster plot
if not spike_times_list:
    print("No spikes found for the selected units in the specified time range. Skipping plot generation.")
else:
    plt.figure(figsize=(15, 8))
    plt.eventplot(spike_times_list, linelengths=0.75, colors='black', lineoffsets=unit_plot_indices)
    plt.yticks(unit_plot_indices, [f'Unit {uid}' for uid in actual_unit_ids_plotted]) # Use actual unit IDs for labels
    plt.xlabel('Time (seconds)')
    plt.ylabel('Unit ID')
    plt.title(f'Raster Plot (First {len(actual_unit_ids_plotted)} Units with Spikes, First {time_limit} seconds)')
    plt.xlim(0, time_limit)
    plt.grid(axis='x', alpha=0.5)
    plt.margins(y=0.02) # Add small margin to y-axis

    plot_path = "explore/units_raster_subset.png"
    plt.savefig(plot_path)
    print(f"Raster plot saved to: {plot_path}")

# Close the file handles
io.close()
print("File handles closed.")