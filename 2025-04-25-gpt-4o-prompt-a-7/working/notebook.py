# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA Disruption with DREADDs
#
# **Disclaimer**: This notebook was AI-generated and has not been fully verified. Users should exercise caution when interpreting the code or results presented here.

# %% [markdown]
# ## Overview of Dandiset 001375
# Dandiset 001375, titled "Septum GABA disruption with DREADDs," is a pilot study exploring the effects of disrupting septal GABAergic activity on hippocampal and neocortical function. The Dandiset is openly accessible and is licensed under CC-BY-4.0.
# 
# More information can be found on the [Dandiset page](https://dandiarchive.org/dandiset/001375).

# %% [markdown]
# ## Notebook Objectives
# This notebook will guide the user through:
# 1. Connecting to the DANDI API and loading the Dandiset metadata.
# 2. Accessing and exploring NWB files within the Dandiset.
# 3. Visualizing data from the NWB files.
# 4. Providing insights and directions for further analysis.

# %% [markdown]
# ## Required Packages
# This notebook assumes the following packages are installed: `dandi`, `pynwb`, `h5py`, `remfile`, and `matplotlib`.

# %% 
# Import necessary packages
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Loading the Dandiset
# 

# Connect to DANDI archive
client = DandiAPIClient()

# Retrieve the Dandiset
dandiset = client.get_dandiset("001375")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata.get('name', 'N/A')}")
print(f"Dandiset URL: {metadata.get('url', 'N/A')}")

# %% [markdown]
# ### Listing the Assets
# The assets in the Dandiset can be listed to identify files of interest.

# 
# List the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"Found {len(assets)} assets in the dataset")
print("First 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading an NWB File
# We'll load one of the NWB files and examine its contents. The file path chosen is `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb`. The asset's URL can be explored further on [NeuroSift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft).

# %% 
# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Display basic session information
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"File creation date: {nwb.file_create_date}")

# %% [markdown]
# ## Visualizing Data
# Here, we will visualize some data from the loaded NWB file.

# 
# Plot a subset of time series data
plt.figure(figsize=(12, 6))
plt.plot(time_series.data[:1000, 0], label='Sample Data')
plt.title('Time Series Sample Plot')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
# This notebook introduced the Dandiset and guided you through the process of accessing and visualizing its contents. Future analysis can focus on specific experiments and advanced data processing techniques.