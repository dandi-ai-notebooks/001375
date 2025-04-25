# Script to explore electrode locations from the NWB file
# https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/
# It generates a scatter plot of electrode x, y coordinates, colored by group.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# Access the electrodes table
electrodes = nwb.electrodes
print(f"Accessed 'electrodes' table with columns: {electrodes.colnames}")
print(f"Total number of electrodes: {len(electrodes.id)}")

# Get electrode data needed for plotting
# Converting to DataFrame is convenient here
electrodes_df = electrodes.to_dataframe()
print("Converted electrodes table to DataFrame.")
print(electrodes_df.head())

# Create and save the scatter plot
plt.figure(figsize=(8, 10))
# Use seaborn for easier coloring by group
sns.scatterplot(data=electrodes_df, x='x', y='y', hue='group_name', s=50, style='group_name')

plt.title('Electrode Locations')
plt.xlabel('X coordinate (micrometers)')
plt.ylabel('Y coordinate (micrometers)')
plt.legend(title='Electrode Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
plt.grid(True, alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

plot_path = "explore/electrode_locations.png"
plt.savefig(plot_path)
print(f"Electrode locations plot saved to: {plot_path}")

# Close the file handles
io.close()
print("File handles closed.")