# This script explores the electrode locations and creates a visualization
# This helps understand the spatial arrangement of the recording probes

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

# Add a numerical group ID for coloring
electrodes_df['group_id'] = electrodes_df['group_name'].apply(lambda x: 0 if x == 'shank1' else 1)

# Create a plot of electrode locations
plt.figure(figsize=(10, 8))
colors = ['blue', 'red']
groups = electrodes_df['group_name'].unique()

for i, group in enumerate(groups):
    group_data = electrodes_df[electrodes_df['group_name'] == group]
    plt.scatter(group_data['x'], group_data['y'], c=colors[i], label=group, alpha=0.7)

plt.xlabel('X position (μm)')
plt.ylabel('Y position (μm)')
plt.title('Electrode locations')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig('explore/electrode_locations.png', dpi=300)
print("Saved electrode locations plot to explore/electrode_locations.png")