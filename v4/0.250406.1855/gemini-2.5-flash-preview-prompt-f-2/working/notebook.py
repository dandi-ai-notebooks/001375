# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA disruption with DREADDs
#
# **Important**: This notebook was AI-generated and has not been fully verified. Please exercise caution when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset (**DANDI:001375/0.250406.1855**) is titled "Septum GABA disruption with DREADDs". It is a pilot study investigating the effect of disrupting septal gabaergic activity using DREADDs on hippocampal and neocortical activity.
#
# You can find the Dandiset here: https://dandiarchive.org/dandiset/001375/0.250406.1855

# %% [markdown]
# ## What this notebook covers
#
# This notebook will demonstrate how to:
# - Load the Dandiset using the DANDI API.
# - Load an NWB file from the Dandiset.
# - Examine the structure and contents of the NWB file.
# - Visualize a subset of the raw electrophysiology data.
# - Examine and visualize the sorted unit (spike) data.

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following packages:
# - dandi
# - pynwb
# - h5py
# - remfile
# - numpy
# - matplotlib
# - pandas

# %% [markdown]
# ## Loading the Dandiset
#
# We can load the Dandiset using the `DandiAPIClient`.

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001375", "0.250406.1855")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB file
#
# We will load the NWB file located at `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb`.
#
# The URL for this asset is: `https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/`
#
# We can use `pynwb`, `h5py`, and `remfile` to stream data directly from the Dandi Archive.

# %%
import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"NWB file identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# ## Contents of the NWB file
#
# This NWB file contains extracellular electrophysiology data, along with trial information and sorted units.
#
# Here is a summary of some key contents:
#
# *   **acquisition**: Raw electrophysiology data
#     *   `timestamps`: Not a dataset, timestamps can be reconstructed from `starting_time` and `rate`.
#     *   `rate`: Sampling rate (e.g., 30000.0 Hz)
#     *   `data`: The raw voltage traces (shape: number of time points x number of channels)
# *   **intervals**: Time intervals for experimental paradigms
#     *   `trials`: Table containing start and stop times for trials.
# *   **electrodes**: Table with information about each electrode.
#     *   Columns include `x`, `y` coordinates, `location`, `filtering`, `group`, `group_name`, and `label`.
# *   **units**: Table containing information about sorted units (neurons).
#     *   Columns include `spike_times`.

# %% [markdown]
# You can explore this NWB file interactively on Neurosift:
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=0.250406.1855

# %% [markdown]
# ## Examining Trial Intervals and Electrodes
#
# We can load the trial information and electrode details into pandas DataFrames for easier inspection.

# %%
import pandas as pd

# Load trials data
trials_df = nwb.trials.to_dataframe()
print("Trials DataFrame Head:")
print(trials_df.head())

# Load electrodes data
electrodes_df = nwb.electrodes.to_dataframe()
print("\nElectrodes DataFrame Head:")
print(electrodes_df.head())

# %% [markdown]
# The `trials` DataFrame shows the start and stop times for different trials. The `electrodes` DataFrame provides spatial and grouping information for each of the recording channels.

# %% [markdown]
# ## Visualizing a subset of Raw Electrophysiology Data
#
# The raw electrophysiology data is quite large, so we will load and visualize a small subset of the data to see the voltage traces.
#
# We will plot the first 4 channels for a 0.1 second interval starting at 10 seconds.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Access the electrophysiology data and properties
ecephys_data = nwb.acquisition['time_series'].data
sampling_rate = nwb.acquisition['time_series'].rate
starting_time = nwb.acquisition['time_series'].starting_time

# Select a time window (10 seconds to 10.1 seconds)
start_time = 10.0
end_time = 10.1
start_index = int(start_time * sampling_rate)
end_index = int(end_time * sampling_rate)

# Select a few channels (first 4)
channel_indices = [0, 1, 2, 3]

# Load the data subset
ecephys_subset = ecephys_data[start_index:end_index, channel_indices]

# Generate timestamps for the subset
timestamps_data = starting_time + np.arange(start_index, end_index) / sampling_rate

# Plot the data
plt.figure(figsize=(10, 6))
for i, channel_index in enumerate(channel_indices):
    # Get the actual electrode label for the legend
    electrode_label = electrodes_df.iloc[channel_index]['label']
    plt.plot(timestamps_data, ecephys_subset[:, i], label=f'{electrode_label}')

plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Raw Electrophysiology Data Subset (Channels 0-3)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# This plot shows the raw voltage fluctuations recorded from a few channels over a short period. This can be useful for assessing the signal quality.

# %% [markdown]
# ## Examining and Visualizing Units Data
#
# The `units` table contains the timestamps of detected spikes for each sorted unit. We can examine the distribution of spike counts across units.

# %%
# Load units data
units_df = nwb.units.to_dataframe()
print("Units DataFrame Head:")
print(units_df.head())

# Calculate spike counts per unit
spike_counts = units_df['spike_times'].apply(len)

# Plot histogram of spike counts
plt.figure(figsize=(10, 6))
plt.hist(spike_counts, bins=20)
plt.xlabel("Number of Spikes")
plt.ylabel("Number of Units")
plt.title("Histogram of Spike Counts per Unit")
plt.grid(True)
plt.show()

# %% [markdown]
# The histogram shows the distribution of the total number of spikes recorded for each sorted unit. We can see that some units are much more active than others.

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to access and explore several key data types within this Dandiset's NWB file, including raw electrophysiology, trial intervals, electrode information, and sorted units. We visualized a small subset of the raw data and examined the distribution of spike counts.
#
# Possible future directions for analysis could include:
# - Analyzing the relationship between neural activity (raw data or spike times) and the defined trials.
# - Investigating the spatial distribution of units and their activity patterns based on electrode locations.
# - Performing more advanced analyses on the spike times, such as inter-spike intervals or cross-correlations between units.
# - Exploring other potential data streams or metadata within the NWB file that were not covered here.

# %%
io.close()