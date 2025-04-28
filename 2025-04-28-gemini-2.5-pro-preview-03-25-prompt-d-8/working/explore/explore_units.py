# Explore units (spike times)
# Load spike times for the first 5 units.
# Create a raster plot to visualize their firing patterns over the first 60 seconds.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print('Loading NWB file...')
# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print('Accessing units data...')
units_df = nwb.units.to_dataframe()
num_units_to_plot = 5
unit_ids = units_df.index[:num_units_to_plot]
max_time = 60.0 # Plot first 60 seconds

print(f'Plotting raster for the first {num_units_to_plot} units up to {max_time} seconds...')
spike_times_list = []
valid_unit_ids = [] # Track units that have spikes within the time range

# Collect spike times within the desired time range
for i, unit_id in enumerate(unit_ids):
    try:
        # Get spike times for the current unit
        st = nwb.units['spike_times'][unit_id][:]
        # Filter spikes within the max_time
        st_filtered = st[st <= max_time]
        if len(st_filtered) > 0:
            spike_times_list.append(st_filtered)
            valid_unit_ids.append(unit_id) # Keep track of the actual unit ID
        else:
             print(f"Unit {unit_id} has no spikes before {max_time}s, skipping.")
    except IndexError:
        print(f"Could not access spike times for unit index {i} (ID: {unit_id}). Skipping.")
    except Exception as e:
        print(f"An error occurred for unit index {i} (ID: {unit_id}): {e}. Skipping.")


if not spike_times_list:
    print("No spike times found for the selected units within the specified time range.")
else:
    # Plot raster using valid units
    sns.set_theme()
    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_times_list, linelengths=0.75)
    plt.yticks(np.arange(len(valid_unit_ids)), valid_unit_ids) # Use actual unit IDs for labels
    plt.xlabel('Time (s)')
    plt.ylabel('Unit ID')
    plt.title(f'Raster Plot of First {len(valid_unit_ids)} Units (0-{max_time}s)')
    plt.xlim(0, max_time)
    plt.ylim(-0.5, len(valid_unit_ids) - 0.5) # Adjust ylim based on the number of plotted units
    plt.savefig('explore/units_raster.png')
    plt.close() # Close the plot to prevent hanging
    print(f'Raster plot saved to explore/units_raster.png for units: {valid_unit_ids}')

print('Script finished.')