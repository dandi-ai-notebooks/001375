# %% [markdown]
# Exploring Dandiset 001375: Septum GABA Disruption with DREADDs

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# This notebook provides an overview of Dandiset 001375, which contains data from a pilot study of the effect of disrupting septal GABAergic activity using DREADDs on hippocampal and neocortical activity.
#
# [Dandiset 001375](https://dandiarchive.org/dandiset/001375)
#
# This notebook will cover the following:
#
# 1.  Loading the Dandiset metadata using the DANDI API.
# 2.  Listing the assets (files) in the Dandiset.
# 3.  Loading an NWB file and exploring its contents.
# 4.  Visualizing example data from the NWB file.

# %% [markdown]
# ### Required Packages
#
# The following packages are required to run this notebook:
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy
# *   seaborn

# %%
# Load the Dandiset using the DANDI API
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001375")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"\nFound {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ### Loading and Exploring an NWB File
#
# In this section, we will load one of the NWB files in the Dandiset and explore its contents.
#
# We will load the following file: `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb`
#
# The URL for this asset is: `https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/`
#
# You can also explore this file on neurosift: [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft)

# %%
import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Print some basic information about the NWB file
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# ### Exploring Electrode Groups
#
# Let's explore the electrode groups in the NWB file.

# %%
# Electrode groups
electrode_groups = nwb.electrode_groups
print(electrode_groups)

# Iterate through the electrode groups and print their descriptions and locations
for name, group in electrode_groups.items():
    print(f"\nElectrode Group: {name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device Description: {group.device.description}")
    print(f"  Device Manufacturer: {group.device.manufacturer}")

# %% [markdown]
# ### Exploring TimeSeries data
#
# Now let's examine the `time_series` object within `nwb.acquisition`.

# %%
# Access the time_series data
acquisition = nwb.acquisition
time_series = acquisition["time_series"]

# Print information about the TimeSeries
print(f"Starting time: {time_series.starting_time}")
print(f"Rate: {time_series.rate}")
print(f"Unit: {time_series.unit}")
print(f"Data shape: {time_series.data.shape}")

# %% [markdown]
# ### Visualizing a subset of the TimeSeries data
#
# It's important to load only a subset of rows and columns from the `time_series.data` dataset, since the entire dataset requires considerable memory.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load a small subset of the data (e.g., the first 1000 rows and the first 10 channels)
num_rows = 1000
num_channels = 10

data_subset = time_series.data[:num_rows, :num_channels]
time = np.arange(0, num_rows / time_series.rate, 1 / time_series.rate)

# Plot the subset of data
plt.figure(figsize=(12, 6))
for i in range(num_channels):
    plt.plot(time, data_subset[:, i] + i * 100, label=f"Channel {i}")  # Offset each channel for better visualization

plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV) + offset")
plt.title("Subset of TimeSeries Data")
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ### Exploring Trials
#
# Now let's explore the trials data within the NWB file.

# %%
# Access the trials data
trials = nwb.trials

# Print information about the trials
print(f"Trials description: {trials.description}")
print(f"Column names: {trials.colnames}")

# Convert the trials data to a Pandas DataFrame
import pandas as pd
trials_df = trials.to_dataframe()

# Print the first 5 rows of the trials DataFrame
print("\nFirst 5 rows of the trials DataFrame:")
print(trials_df.head())

# %% [markdown]
# ### Exploring Units
#
# Now let's explore the units data within the NWB file.

# %%
# Access the units data
units = nwb.units

# Print information about the units
print(f"Units description: {units.description}")
print(f"Column names: {units.colnames}")

# Convert the units data to a Pandas DataFrame
units_df = units.to_dataframe()

# Print the first 5 rows of the units DataFrame
print("\nFirst 5 rows of the units DataFrame:")
print(units_df.head())

# %% [markdown]
# ### Exploring the spike times for the first unit
#
# It is important to avoid loading all spike times into memory at once.
# Instead, access spike times of individual units.

# %%
# Access the spike times for the first unit
if len(units_df) > 0:
    first_unit_id = units_df.index[0]
    spike_times = units_df['spike_times'][first_unit_id]
    print(f"The spike times of the first unit {first_unit_id}:")
    print(spike_times)
else:
    print("No units found in the NWB file.")

# %% [markdown]
# ### Summary and Future Directions
#
# This notebook has provided a basic introduction to exploring Dandiset 001375. We have shown how to:
#
# 1.  Load the Dandiset metadata using the DANDI API.
# 2.  List the assets (files) in the Dandiset.
# 3.  Load an NWB file and explore its contents.
# 4.  Visualize example data from the NWB file.
#
# Possible future directions for analysis include:
#
# *   Performing more in-depth analysis of the electrophysiology data.
# *   Comparing data across different sessions or subjects.
# *   Investigating the effect of septal GABA disruption on hippocampal and neocortical activity.