# This script creates an improved visualization of electrode locations
# with more appropriate colors and markers to distinguish the shanks

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print("Loading NWB file from:", url)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrodes data
electrodes_df = nwb.electrodes.to_dataframe()

# Create a scatter plot with different markers for each shank
plt.figure(figsize=(10, 8))

shank1_df = electrodes_df[electrodes_df['group_name'] == 'shank1']
shank2_df = electrodes_df[electrodes_df['group_name'] == 'shank2']

# Plot with distinct markers and colors
plt.scatter(shank1_df['x'], shank1_df['y'], c='blue', marker='o', 
            label='Shank 1', alpha=0.7, s=30)
plt.scatter(shank2_df['x'], shank2_df['y'], c='red', marker='s', 
            label='Shank 2', alpha=0.7, s=30)

plt.xlabel('X position (μm)')
plt.ylabel('Y position (μm)')
plt.title('Electrode locations by shank')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Add horizontal jitter to distinguish overlapping points
plt.xlim(-25, 25)

# Save the plot
plt.tight_layout()
plt.savefig('explore/electrode_locations_improved.png', dpi=300)
print("Saved improved electrode locations plot to explore/electrode_locations_improved.png")

# Create an additional plot to visualize electrode locations by their x positions
# This helps see the probe geometry more clearly
plt.figure(figsize=(12, 8))

# Group by x position and group_name, count the number of electrodes
pos_counts = electrodes_df.groupby(['x', 'group_name']).size().reset_index(name='count')

# Plot as a bar chart
x_positions = sorted(pos_counts['x'].unique())
bar_width = 0.35

for i, shank in enumerate(['shank1', 'shank2']):
    shank_data = pos_counts[pos_counts['group_name'] == shank]
    
    # Create a bar chart showing electrode count at each x position
    bars = plt.bar([p + bar_width*i for p in range(len(x_positions))], 
                  [shank_data[shank_data['x'] == pos]['count'].values[0] if not shank_data[shank_data['x'] == pos].empty else 0 
                   for pos in x_positions],
                  bar_width, label=f'Shank {i+1}',
                  color='blue' if i == 0 else 'red')

plt.xlabel('X position')
plt.ylabel('Number of electrodes')
plt.title('Electrode distribution by x-coordinate')
plt.xticks([p + bar_width/2 for p in range(len(x_positions))], [f"{p:g}" for p in x_positions])
plt.legend()

# Save the count plot
plt.tight_layout()
plt.savefig('explore/electrode_positions_counts.png', dpi=300)
print("Saved electrode position counts to explore/electrode_positions_counts.png")