# Explore spike times from the Units table
# Goal: Load spike times for a few units and create a raster plot
# to visualize firing patterns over a time interval.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Ensure the output directory exists
os.makedirs('explore', exist_ok=True)

print("Loading NWB file...")
# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Explicitly read-only
nwb = io.read()
print("NWB file loaded.")

# Access the units table
units_df = nwb.units.to_dataframe()
print(f"Units table loaded with {len(units_df)} units.")
print("Units table head:")
print(units_df.head())

# Define units and time range
num_units_to_plot = min(5, len(units_df)) # Plot first 5 units or fewer if less exist
unit_ids = units_df.index[:num_units_to_plot]
time_limit_seconds = 60.0 # Plot spikes within the first 60 seconds

print(f"Preparing spike times for units {list(unit_ids)} up to {time_limit_seconds}s...")
spike_times_list = []
actual_unit_ids_plotted = []

# Iterate through selected unit IDs and get spike times within the time limit
for unit_id in unit_ids:
    # Access spike times for the current unit
    # The spike_times column contains arrays of spike times for each unit
    unit_spike_times = units_df.loc[unit_id, 'spike_times']

    # Filter spike times within the specified time limit
    filtered_spike_times = unit_spike_times[unit_spike_times <= time_limit_seconds]

    if len(filtered_spike_times) > 0:
        spike_times_list.append(filtered_spike_times)
        actual_unit_ids_plotted.append(unit_id)
    else:
        print(f"Unit {unit_id} has no spikes before {time_limit_seconds}s. Skipping.")


if not spike_times_list:
    print("No spike times found for the selected units in the specified time range. Cannot generate plot.")
else:
    print(f"Generating raster plot for {len(spike_times_list)} units...")
    # Use seaborn styling
    sns.set_theme()

    # Plot the raster
    fig, ax = plt.subplots(figsize=(12, 4))
    # Use eventplot for raster plots
    ax.eventplot(spike_times_list, linelengths=0.75, colors='black')

    ax.set_xlabel('Time (s)')
    ax.set_yticks(range(len(actual_unit_ids_plotted))) # Positions for y-ticks
    ax.set_yticklabels(actual_unit_ids_plotted) # Label y-ticks with actual unit IDs
    ax.set_ylabel('Unit ID')
    ax.set_title(f'Spike Raster Plot (First {len(actual_unit_ids_plotted)} Units, First {time_limit_seconds}s)')
    ax.set_xlim(0, time_limit_seconds)
    ax.grid(axis='x', linestyle='--', alpha=0.7) # Add vertical grid lines for time reference

    # Save the plot
    output_path = 'explore/spike_raster.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

# Close resources
io.close()
# remote_file.close()

print("Script finished.")