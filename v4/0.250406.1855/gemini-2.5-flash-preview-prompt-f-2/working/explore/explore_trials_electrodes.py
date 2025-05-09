# This script explores the trials and electrodes data in the NWB file.

import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Explore trials
trials_df = nwb.trials.to_dataframe()
print("Trials DataFrame Head:")
print(trials_df.head())

# Explore electrodes
electrodes_df = nwb.electrodes.to_dataframe()
print("\nElectrodes DataFrame Head:")
print(electrodes_df.head())

io.close()