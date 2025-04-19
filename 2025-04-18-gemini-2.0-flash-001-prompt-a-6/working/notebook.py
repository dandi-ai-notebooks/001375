# %% [markdown]
# Exploring Dandiset 001375: Septum GABA Disruption with DREADDs

# %% [markdown]
# **Important:** This notebook was AI-generated and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# This notebook explores Dandiset 001375, which contains data from a pilot study of the effect of disrupting septal GABAergic activity using DREADDs on hippocampal and neocortical activity.
#
# Link to the Dandiset: https://dandiarchive.org/dandiset/001375

# %% [markdown]
# The notebook will cover:
#
# 1.  Loading the Dandiset using the DANDI API.
# 2.  Loading an NWB file and exploring its metadata.
# 3.  Loading and visualizing electrophysiology data from the NWB file.
#

# %% [markdown]
# Required packages:
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy
# *   seaborn

# %%
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
# Loading an NWB file and exploring its metadata.
#
# We will load the following NWB file: `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb`
#
# The URL for the asset is: `https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/`

# %%
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb.session_description # (str) mouse running laps in virtual hallway
nwb.identifier # (str) MS13B
nwb.session_start_time # (datetime) 2024-07-25T19:00:00-07:00
nwb.timestamps_reference_time # (datetime) 2024-07-25T19:00:00-07:00
nwb.file_create_date # (list) \[datetime.datetime(2025, 4, 5, 16, 50, 15, 663983, tzinfo=tzoffset(None, -25200))]
nwb.acquisition # (LabelledDict)

# %% [markdown]
# Loading and visualizing electrophysiology data from the NWB file.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# %%
nwb.acquisition["time_series"].starting_time # (float64) 0.0
nwb.acquisition["time_series"].rate # (float64) 30000.0
nwb.acquisition["time_series"].resolution # (float64) -1.0
nwb.acquisition["time_series"].comments # (str) no comments
nwb.acquisition["time_series"].description # (str) no description
nwb.acquisition["time_series"].conversion # (float64) 1.0
nwb.acquisition["time_series"].offset # (float64) 0.0
nwb.acquisition["time_series"].unit # (str) mV
nwb.acquisition["time_series"].data # (Dataset) shape (144675584, 256); dtype int16

# %% [markdown]
# Let's load a small subset of the data and plot it. Note that the data is streamed from the server, so it may take a few seconds to load.

# %%
data = nwb.acquisition["time_series"].data[0:1000, 0:10]
plt.figure(figsize=(10, 5))
plt.plot(data)
plt.xlabel("Time (samples)")
plt.ylabel("Voltage (mV)")
plt.title("Electrophysiology Data")
plt.show()

# %% [markdown]
# Now let's look at the electrodes table.

# %%
nwb.electrodes.colnames

# %%
nwb.electrodes.to_dataframe().head()

# %% [markdown]
# Now let's look at the units table.

# %%
nwb.units.colnames

# %%
nwb.units.to_dataframe().head()

# %% [markdown]
# Let's load the spike times for the first unit and plot them.

# %%
spike_times = nwb.units.spike_times[0:100]
plt.figure(figsize=(10, 5))
plt.plot(spike_times, np.zeros_like(spike_times), "|")
plt.xlabel("Time (s)")
plt.ylabel("Unit")
plt.title("Spike Times for Unit 0")
plt.show()

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how to load and explore data from a DANDI archive Dandiset. We showed how to load the Dandiset, load an NWB file, and visualize electrophysiology data.
#
# ## Possible future directions
#
# *   Explore other NWB files in the Dandiset.
# *   Explore other data modalities in the NWB file such as stimulus information from the trials table.
# *   Perform more advanced analysis of the electrophysiology data such as spike sorting.