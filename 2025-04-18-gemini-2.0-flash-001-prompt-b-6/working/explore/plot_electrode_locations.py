# %%
# This script loads electrode locations from an NWB file and plots them.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode locations
electrode_x = nwb.electrodes.x[:]
electrode_y = nwb.electrodes.y[:]

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(electrode_x, electrode_y)
plt.xlabel("Electrode X Location")
plt.ylabel("Electrode Y Location")
plt.title("Electrode Locations")
plt.savefig("explore/electrode_locations.png")
plt.close()