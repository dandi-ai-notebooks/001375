import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# This script loads the electrode locations from the NWB file and plots them.
# The plot is saved as a PNG file in the explore/ directory.

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Extract electrode locations
electrodes = nwb.electrodes.to_dataframe()
x = electrodes['x']
y = electrodes['y']

# Plot the electrode locations
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.xlabel("X Location")
plt.ylabel("Y Location")
plt.title("Electrode Locations")
plt.savefig("explore/electrode_locations.png")
plt.close()