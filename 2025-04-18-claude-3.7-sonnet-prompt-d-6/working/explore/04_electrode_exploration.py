#!/usr/bin/env python
"""
This script explores the electrode information in the NWB file.
We'll visualize the spatial arrangement and properties of the electrodes.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrodes information
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Number of electrodes: {len(electrodes_df)}")

# Print summary of electrode groups
print("\nElectrode groups summary:")
print(electrodes_df['group_name'].value_counts())

# Print summary of electrode locations
print("\nElectrode locations:")
if 'location' in electrodes_df.columns:
    print(electrodes_df['location'].value_counts())
else:
    print("No location information available")

# Examine x and y coordinates to understand the probe geometry
print("\nX coordinates range:", electrodes_df['x'].min(), "to", electrodes_df['x'].max())
print("Y coordinates range:", electrodes_df['y'].min(), "to", electrodes_df['y'].max())

# Plot electrode positions
plt.figure(figsize=(10, 10))
for group_name, group_df in electrodes_df.groupby('group_name'):
    plt.scatter(group_df['x'], group_df['y'], label=group_name, alpha=0.7)
plt.xlabel('X position (μm)')
plt.ylabel('Y position (μm)')
plt.title('Electrode Positions')
plt.legend()
plt.grid(True)
plt.savefig('explore/electrode_positions.png')

# Create a more detailed view of each shank separately
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
for i, (name, group_df) in enumerate(electrodes_df.groupby('group_name')):
    axs[i].scatter(group_df['x'], group_df['y'])
    axs[i].set_title(f'Electrode positions for {name}')
    axs[i].set_xlabel('X position (μm)')
    axs[i].set_ylabel('Y position (μm)')
    axs[i].grid(True)
    
    # Annotate some points with their electrode numbers
    for j in range(0, len(group_df), 10):  # Label every 10th electrode
        electrode_id = group_df.index[j]
        axs[i].annotate(str(electrode_id), 
                       (group_df['x'].iloc[j], group_df['y'].iloc[j]),
                       textcoords="offset points", 
                       xytext=(0,5), 
                       ha='center')
plt.tight_layout()
plt.savefig('explore/detailed_electrode_positions.png')

# Plot the filtering and other properties if available
if 'filtering' in electrodes_df.columns:
    print("\nFiltering information:", electrodes_df['filtering'].unique())

# Close the file handles
h5_file.close()
remote_file.close()