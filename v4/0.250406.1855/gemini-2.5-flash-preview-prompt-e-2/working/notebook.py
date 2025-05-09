# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA disruption with DREADDs
#
# **Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of Dandiset 001375
#
# This Dandiset, titled "Septum GABA disruption with DREADDs", contains data from a pilot study investigating the effect of disrupting septal gabaergic activity using DREADDs on hippocampal and neocortical activity in mice.
#
# **Dandiset URL:** https://dandiarchive.org/dandiset/001375/0.250406.1855

# %% [markdown]
# ## Notebook Contents
#
# This notebook will demonstrate how to:
# - Access basic information about the Dandiset using the DANDI API.
# - List the assets (files) contained within the Dandiset.
# - Load a specific NWB file from the Dandiset.
# - Explore the structure and contents of the NWB file.
# - Load and visualize some sample data from the NWB file.

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following Python packages:
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `itertools`
# - `seaborn`

# %%
# Import necessary packages
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up seaborn theme for better looking plots
sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset and Listing Assets
#
# We will connect to the DANDI archive and load the specified Dandiset. Then we will list the first few assets to get an idea of the files available.

# %%
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
# ## Loading an NWB File
#
# We will now load one of the NWB files from the Dandiset to examine its structure and contents. We will load the file located at `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb` using its direct download URL.

# %%
# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Display some basic metadata from the NWB file
print(f"NWB file session description: {nwb.session_description}")
print(f"NWB file identifier: {nwb.identifier}")
print(f"NWB file session start time: {nwb.session_start_time}")

# %% [markdown]
# ## NWB File Contents Summary
#
# The loaded NWB file `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb` contains various types of neurophysiology data and associated metadata. Key sections include:
#
# *   **acquisition/time_series**: Raw extracellular electrophysiology recordings. This dataset has a shape of `(144675584, 256)`, indicating a large number of time samples across 256 channels. The data type is `int16` and the unit is `mV` with a sampling rate of `30000.0` Hz.
# *   **electrode_groups**: Information about groups of electrodes, such as shanks. This file contains information for `shank1` and `shank2`, both described as 128-channel silicon probes with locations in the ventral hippocampus and visual cortex.
# *   **devices**: Details about the recording devices used. This file describes a "silicon probe array" manufactured by UCLA with 128 channels.
# *   **intervals/trials**: Time intervals marking the start and stop times of experimental trials (laps in a virtual hallway). This table has columns for `start_time` and `stop_time` and contains 387 entries.
# *   **electrodes**: A dynamic table providing details for each electrode, including `x`, `y` coordinates, `location`, `filtering`, `group`, `group_name`, and `label`. This table contains information for all 256 electrodes.
# *   **subject**: Metadata about the experimental subject, including age, sex, species, and description. The subject ID is `MS13B`, a `Mus musculus` male at P90D with a medial septum DREADD manipulation.
# *   **units**: Information about identified single units (neurons), including spike times. This table contains 33 units, each with associated spike times.
#
# You can explore this NWB file interactively on NeuroSift: [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft)

# %% [markdown]
# ## Exploring Electrode Data
#
# We can access the `electrodes` table to see the configuration and details of the recording electrodes.

# %%
# Get the electrodes table as a pandas DataFrame
electrodes_df = nwb.electrodes.to_dataframe()

# Display the first few rows of the electrodes table
print("First 5 rows of the electrodes table:")
print(electrodes_df.head())

# Print the columns of the electrodes table
print("\nColumns in the electrodes table:")
print(electrodes_df.columns.tolist())

# Get electrode locations
electrode_locations = electrodes_df['location'].unique()
print("\nUnique electrode locations:")
print(electrode_locations)

# %% [markdown]
# ## Visualizing Electrophysiology Data
#
# We will load a small subset of the raw extracellular electrophysiology data from the `acquisition/time_series` to visualize it. Loading the entire dataset (`shape (144675584, 256)`) would be too much data to load into memory in a notebook.

# %%
# Access the time series data
time_series_data = nwb.acquisition['time_series'].data

# Get the sampling rate and starting time
sampling_rate = nwb.acquisition['time_series'].rate
starting_time = nwb.acquisition['time_series'].starting_time

# Load a small segment of data (e.g., the first 1 second) for visualization
# Calculate the number of samples for 1 second
num_samples_to_load = int(sampling_rate * 1)

# Load data for the first few channels
num_channels_to_plot = 5
data_subset = time_series_data[0:num_samples_to_load, 0:num_channels_to_plot]

# Create a time vector for the loaded data subset
time_subset = starting_time + np.arange(num_samples_to_load) / sampling_rate

# Plot the data subset
plt.figure(figsize=(12, 6))
for i in range(num_channels_to_plot):
    # Offset the traces for visibility
    plt.plot(time_subset, data_subset[:, i] + i * 200, label=f'Channel {i}')

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (offset for clarity)")
plt.title(f"Subset of Raw Electrophysiology Data (First {num_samples_to_load} samples, {num_channels_to_plot} channels)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# **Note:** The plot above shows a small segment of raw voltage traces from a few channels. This is primarily to demonstrate how to load and visualize the data. Further processing like filtering or spike sorting would be required for analyzing neuronal activity.

# %% [markdown]
# ## Exploring Unit Data
#
# The `units` table contains information about the single units (neurons) that were detected and sorted. This includes the spike times for each unit.

# %%
# Get the units table as a pandas DataFrame
units_df = nwb.units.to_dataframe()

# Display the first few rows of the units table
print("First 5 rows of the units table:")
print(units_df.head())

# Print the columns of the units table
print("\nColumns in the units table:")
print(units_df.columns.tolist())

# Print the total number of units
print(f"\nTotal number of units: {len(units_df)}")

# Get the IDs of the first few units
first_unit_ids = units_df.index.tolist()[:5]
print(f"\nIDs of the first 5 units: {first_unit_ids}")


# %% [markdown]
# ## Visualizing Unit Spike Times
#
# We can visualize the spike times for a few selected units to see their activity patterns over a period of time.

# %%
# Select a few unit IDs to visualize
unit_ids_to_plot = units_df.index.tolist()[0:5] # Visualize the first 5 units

plt.figure(figsize=(12, 6))

# Iterate over the selected units and plot their spike times
for i, unit_id in enumerate(unit_ids_to_plot):
    # Get spike times for the current unit
    spike_times = nwb.units['spike_times'][nwb.units.id[:] == unit_id][0]

    # Plot spike times as vertical lines (raster plot like)
    plt.vlines(spike_times, i + 0.5, i + 1.5, color='k', alpha=0.8)

plt.xlabel("Time (seconds)")
plt.ylabel("Unit ID (Arbitrary Index)")
plt.title(f"Spike Times for Selected Units ({', '.join(map(str, unit_ids_to_plot))})")
plt.yticks(np.arange(len(unit_ids_to_plot)) + 1, unit_ids_to_plot)
plt.ylim(0.5, len(unit_ids_to_plot) + 0.5)
plt.grid(True, axis='x')
plt.show()

# %% [markdown]
# **Note:** The plot above shows the occurrences of spikes for a few units as vertical lines over time. This provides a basic visualization of neuronal firing patterns.

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook provided a basic introduction to accessing and exploring data within Dandiset 001375. We demonstrated how to load the Dandiset and an NWB file, examine its structure and metadata, and visualize subsets of raw electrophysiology data and unit spike times.
#
# For further analysis, researchers could:
# - Load and examine data from other NWB files in the Dandiset.
# - Perform more sophisticated analysis on the electrophysiology data, such as filtering, spike sorting (outside the scope of a simple notebook), and calculating power spectral densities.
# - Analyze the relationship between neuronal activity (spike times) and behavioral events (defined in the `intervals/trials` table).
# - Investigate the spatial organization of electrodes and units based on the information in the `electrodes` table.
# - Explore other data modalities if present in other NWB files within the Dandiset.

# %%
# Close the NWB file
io.close()
h5_file.close()