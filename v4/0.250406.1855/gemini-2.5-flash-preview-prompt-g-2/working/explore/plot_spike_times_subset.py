# This script loads spike times for a subset of units and plots them.

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

# Get units data
units_df = nwb.units.to_dataframe()

# Select the first few units
num_units_to_plot = min(5, len(units_df)) # Plot at most 5 units
selected_units = units_df.iloc[0:num_units_to_plot]

# Plot spike times for selected units
plt.figure(figsize=(12, 6))
for index, row in selected_units.iterrows():
    spike_times = row['spike_times']
    # Plot spike times as vertical lines
    plt.vlines(spike_times, index, index + 0.8, label=f'Unit {row.name}', color=sns.color_palette()[index % len(sns.color_palette())])

plt.xlabel('Time (s)')
plt.ylabel('Unit ID')
plt.title(f'Spike Times for First {num_units_to_plot} Units')
plt.yticks(np.arange(num_units_to_plot) + 0.4, selected_units.index) # Center y-ticks between lines
plt.ylim(-0.1, num_units_to_plot + 0.1)
plt.grid(True)
plt.savefig('explore/spike_times_subset.png')
plt.close()