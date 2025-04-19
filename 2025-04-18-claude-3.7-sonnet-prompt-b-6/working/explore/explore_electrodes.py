"""
This script explores the electrode positions and properties in the MS13B NWB file.
The goal is to understand the recording setup and potentially visualize the electrode arrangement.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()

# Print basic information about electrodes
print(f"Total number of electrodes: {len(electrodes_df)}")
print("\nUnique electrode groups:")
print(electrodes_df['group_name'].unique())

print("\nElectrode locations:")
print(electrodes_df['location'].unique())

# Count electrodes by group
print("\nNumber of electrodes by group:")
print(electrodes_df.groupby('group_name').size())

# Plot electrode positions
plt.figure(figsize=(10, 8))
for group_name, group_df in electrodes_df.groupby('group_name'):
    plt.scatter(group_df['x'], group_df['y'], label=group_name, alpha=0.7)

plt.title('Electrode Positions')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.grid(True)
plt.savefig('explore/electrode_positions.png')

# Create a 2D grid visualization showing approximate spatial arrangement
# Assuming electrodes are arranged in a grid pattern
plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
for group_name, group_df in electrodes_df.groupby('group_name'):
    x_vals = group_df['x'].values
    y_vals = group_df['y'].values
    plt.scatter(x_vals, y_vals, label=group_name, alpha=0.7)
    
    # Add electrode indices to the plot for a few electrodes
    for i, (x, y, label) in enumerate(zip(x_vals, y_vals, group_df['label'])):
        if i % 20 == 0:  # Label every 20th electrode to avoid cluttering
            plt.text(x, y, label, fontsize=8)

plt.title('Electrode Positions with Selected Labels')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.grid(True)

# Create a heatmap-style visualization of electrode positions
plt.subplot(1, 2, 2)
x_min, x_max = electrodes_df['x'].min(), electrodes_df['x'].max()
y_min, y_max = electrodes_df['y'].min(), electrodes_df['y'].max()
x_range = x_max - x_min
y_range = y_max - y_min
grid_size = 20
grid = np.zeros((grid_size, grid_size))

for _, row in electrodes_df.iterrows():
    x_idx = int((row['x'] - x_min) / x_range * (grid_size - 1))
    y_idx = int((row['y'] - y_min) / y_range * (grid_size - 1))
    grid[y_idx, x_idx] += 1

plt.imshow(grid, cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar(label='Electrode count')
plt.title('Electrode Density Map')
plt.xlabel('X position (binned)')
plt.ylabel('Y position (binned)')

plt.tight_layout()
plt.savefig('explore/electrode_visualization.png')

# Show information about the electrode groups
print("\nElectrode group details:")
for group_name, group in nwb.electrode_groups.items():
    print(f"\nGroup: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device.description} ({group.device.manufacturer})")

# Close the file
h5_file.close()