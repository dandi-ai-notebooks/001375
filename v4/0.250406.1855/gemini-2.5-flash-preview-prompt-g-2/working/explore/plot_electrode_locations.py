# This script loads the electrodes table and plots electrode locations.

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

# Get electrodes data
electrodes_df = nwb.electrodes.to_dataframe()

# Plot electrode locations, colored by group
plt.figure(figsize=(8, 8))
sns.scatterplot(data=electrodes_df, x='x', y='y', hue='group_name', s=50)
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Electrode Locations by Group')
plt.grid(True)
plt.axis('equal') # Ensure equal scaling for x and y axes
plt.savefig('explore/electrode_locations.png')
plt.close()