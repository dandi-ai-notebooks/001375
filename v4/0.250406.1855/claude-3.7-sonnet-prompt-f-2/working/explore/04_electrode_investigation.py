# This script investigates the electrode grouping to understand 
# why there's an unexpected imbalance in the visualization

import pynwb
import h5py
import remfile
import pandas as pd

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print("Loading NWB file from:", url)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrodes data
electrodes_df = nwb.electrodes.to_dataframe()

# Count electrodes by group
group_counts = electrodes_df['group_name'].value_counts()
print("\nElectrode counts by group:")
print(group_counts)

# Check unique x coordinates by group
shank1_df = electrodes_df[electrodes_df['group_name'] == 'shank1']
shank2_df = electrodes_df[electrodes_df['group_name'] == 'shank2']

print("\nShank1 unique x coordinates and counts:")
print(shank1_df['x'].value_counts())

print("\nShank2 unique x coordinates and counts:")
print(shank2_df['x'].value_counts())

# Print some samples from each group
print("\nSample from shank1 electrodes:")
print(shank1_df.head())

print("\nSample from shank2 electrodes:")
print(shank2_df.head())

# Check if the group column matches the group_name column
check_df = electrodes_df.copy()
check_df['group_object'] = check_df['group'].apply(lambda x: x.name if hasattr(x, 'name') else 'unknown')
print("\nComparing group object names with group_name column:")
match_count = (check_df['group_object'] == check_df['group_name']).sum()
print(f"Matching group names: {match_count} out of {len(check_df)}")