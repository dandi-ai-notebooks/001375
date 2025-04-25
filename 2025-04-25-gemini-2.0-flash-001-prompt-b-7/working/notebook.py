# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA disruption with DREADDs

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# This notebook provides an overview of Dandiset 001375, which contains data from a pilot study of the effect of disrupting septal gabaergic activity using DREADDs on hippocampal and neocortical activity. The Dandiset can be found at [https://dandiarchive.org/dandiset/001375](https://dandiarchive.org/dandiset/001375).
#
# This notebook will cover the following:
#
# 1.  Loading the Dandiset using the DANDI API.
# 2.  Loading and visualizing data from an NWB file within the Dandiset.

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook:
#
# *   `pynwb`
# *   `h5py`
# *   `remfile`
# *   `matplotlib`
# *   `numpy`

# %% [markdown]
# ## Loading the Dandiset
#
# The following code shows how to load the Dandiset using the DANDI API:

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
# ## Loading and Visualizing Data from an NWB File
#
# The following code shows how to load data from one of the NWB files in the Dandiset.
# We will load the file "sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb".

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

# %% [markdown]
# For interactive exploration of this NWB file in your browser, see this link:
#
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft)

# %% [markdown]
# ### Visualizing a segment of time_series.data

# %%
import matplotlib.pyplot as plt
import numpy as np

# Access time_series data
time_series = nwb.acquisition["time_series"]
data = time_series.data
rate = time_series.rate

# Select a segment of data (e.g., first 10 seconds, first channel)
duration = 10  # seconds
start_time = 0  # seconds
start_index = int(start_time * rate)
end_index = int((start_time + duration) * rate)
channel_index = 0
segment = data[start_index:end_index, channel_index]
time = np.arange(start_time, start_time + duration, 1/rate)

# Plot the segment
plt.figure(figsize=(10, 5))
plt.plot(time, segment)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Segment of time_series.data")
plt.show()

# %% [markdown]
# ### Visualizing Trials: Start Time vs. Stop Time

# %%
# Access trials data
trials = nwb.trials
start_time = trials.start_time[:]
stop_time = trials.stop_time[:]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(start_time, stop_time)
plt.xlabel("Start Time (s)")
plt.ylabel("Stop Time (s)")
plt.title("Trials: Start Time vs. Stop Time")
plt.show()

# %% [markdown]
# ### Visualizing distribution of spike times across units

# %%
# Access units data
units = nwb.units
spike_times = units.spike_times[:]

# Create histogram of spike times
plt.figure(figsize=(10, 6))
plt.hist(spike_times, bins=50)
plt.xlabel("Spike Times (s)")
plt.ylabel("Number of Spikes")
plt.title("Distribution of Spike Times across Units")
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook provided a basic overview of how to load and visualize data from Dandiset 001375. Possible future directions for analysis include:
#
# *   Exploring the relationship between neural activity and behavior.
# *   Performing more advanced signal processing on the electrophysiology data.
# *   Comparing data across different subjects or experimental conditions.