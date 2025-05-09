# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA disruption with DREADDs
#
# **AI-generated notebook – caution: not fully verified.**  
# This notebook was generated with the help of AI and is intended as a starting point for exploring Dandiset 001375 (version 0.250406.1855) from the DANDI Archive. Please **read code and results carefully** before drawing scientific conclusions.
#
# ---
#
# ## Dandiset Overview
#
# - **Title:** Septum GABA disruption with DREADDs
# - **Description:** Pilot study of the effect of disrupting septal gabaergic activity using DREADDs on hippocampal and neocortical activity.
# - **Contributors:** Eckert, Michael; McNaughton, Bruce; Ferbinteanu, Janina; NIH Brain
# - **License:** CC-BY-4.0
# - **Version:** 0.250406.1855
# - **Dandiset link:** [https://dandiarchive.org/dandiset/001375/0.250406.1855](https://dandiarchive.org/dandiset/001375/0.250406.1855)
#
# ## What this notebook covers
#
# - Loading of Dandiset metadata and sample NWB file using the DANDI API
# - Examination of the available data and its organization in NWB
# - Visualization of selected data and metadata
# - Suggestions for further exploration

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following packages (install them if needed):
# - dandi
# - pynwb
# - remfile
# - h5py
# - numpy
# - pandas
# - matplotlib
# - seaborn

# %% [markdown]
# ## Connect to the DANDI Archive and load Dandiset metadata

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("001375", "0.250406.1855")

metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Select an NWB file for exploration
#
# For illustration, we'll use the file:
# - **Path:** `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb`
# - **DANDI asset ID:** `ce525828-8534-4b56-9e47-d2a34d1aa897`
# - **Download URL:**  
#   https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/
#
# You can also open this file on Neurosift for interactive exploration:  
# [Neurosift NWB Viewer](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft)
#
# We'll demonstrate how to connect, inspect, and visualize data from this file below.

# %% [markdown]
# ## Load the NWB file using PyNWB and remfile (streamed access)
#
# > **Note:** Streaming remote NWB files can be slow, especially for large datasets. For quick inspection, we load only small portions as examples.

# %%
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("File loaded.")
print("Session description:", nwb.session_description)
print("Subject:", nwb.subject.subject_id, "-", nwb.subject.description)

# %% [markdown]
# ## NWB file contents overview
#
# ```
# Session description: mouse running laps in virtual hallway
# Session start time: 2024-07-25T19:00:00-07:00
# Subject ID: MS13B
# Subject description: medial septum DREADD to suppress GABA interneurons
#
# Main sections:
# ├─ acquisition
# │   └─ time_series: Raw extracellular recording (shape: [144,675,584 x 256], unit: mV)
# ├─ electrode_groups
# │   ├─ shank1: ventral hippocampus, visual cortex, device: 128 ch silicon probe (UCLA)
# │   └─ shank2: ventral hippocampus, visual cortex, device: 128 ch silicon probe (UCLA)
# ├─ devices
# │   └─ silicon probe array: 128 ch silicon probe (UCLA)
# ├─ intervals
# │   └─ trials (387 laps): start/stop times
# ├─ electrodes: Table of 256 electrodes (x/y/location/filtering/group/group_name/label)
# ├─ units: Table of 33 sorted units (spike_times)
# ```
#
# The table below summarizes some of the core data groups:

# %%
import pandas as pd

summary = pd.DataFrame([
    ["acquisition/time_series", "Raw electrophysiology data", "144,675,584 x 256", "int16, mV"],
    ["electrodes", "Electrode table", "256 x 7", "Varied"],
    ["units", "Sorted units", "33 x 1", "spike_times (variable-length)"],
    ["intervals/trials", "Behavioral laps/trials", "387 x 2", "start_time, stop_time"],
], columns=["Group/Field", "Description", "Shape/Count", "Notes"])
summary

# %% [markdown]
# ## Exploring subject metadata

# %%
subject = nwb.subject
print("Subject ID:", subject.subject_id)
print("Species:", subject.species)
print("Sex:", subject.sex)
print("Age:", subject.age, "(", getattr(subject, 'age__reference', None), ")")
print("Description:", subject.description)

# %% [markdown]
# ## Inspecting time intervals (trials)
#
# Trials mark laps/trials in the experiment. Let's look at the first few laps:

# %%
trials_df = nwb.trials.to_dataframe()
display(trials_df.head())

print(f"Total number of trials: {len(trials_df)}")

# %% [markdown]
# ## Electrode information
#
# The electrodes table lists all electrodes and their metadata. We'll preview the first few rows.

# %%
electrodes_df = nwb.electrodes.to_dataframe()
display(electrodes_df.head())

print(f"Number of electrodes: {len(electrodes_df)}")
print("Electrodes fields:", list(electrodes_df.columns))

# %% [markdown]
# ## Sorted units: spike times
#
# The units table contains information about sorted units (putative neurons). Spike times are variable-length arrays per unit.
# Let's preview the spike times for a few units, and plot a simple raster for the first 5 units.

# %%
units_df = nwb.units.to_dataframe()
display(units_df.head())

print(f"Number of units: {len(units_df)}")

# Plot spike times raster for a few units
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure(figsize=(8, 3))
n_show = 5
for i in range(min(n_show, len(units_df))):
    st = units_df.iloc[i]['spike_times']
    plt.vlines(st, i + 0.5, i + 1.5)
plt.yticks(np.arange(1, n_show + 1), units_df.index[:n_show])
plt.xlabel('Time (s)')
plt.ylabel('Unit ID')
plt.title('Raster: Spike times for first few units')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Raw extracellular data: preview a small segment
#
# The raw data are very large. We'll extract and plot a short snippet (e.g., 1 second from the first 5 channels).
# **Note:** This may still take several seconds due to remote streaming.

# %%
ts = nwb.acquisition["time_series"]
rate = ts.rate
n_chans = ts.data.shape[1]
n_samples = int(rate)  # 1 second

snippet = ts.data[:n_samples, :5]  # first 1 second, first 5 channels
time_axis = np.arange(n_samples) / rate

plt.figure(figsize=(10, 4))
offsets = np.arange(5) * 500
for i in range(5):
    plt.plot(time_axis, snippet[:, i] + offsets[i], label=f'Ch {i}')
plt.xlabel('Time (s)')
plt.title('Raw Ephys Snippet (first 5 channels, 1 second)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Suggestions for further exploration
# - Browse the [Neurosift NWB Viewer](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=draft) for interactive visualization.
# - Explore trial-averaged firing rates, spike time autocorrelations, or LFP signals.
# - Investigate relationships between behavioral events (trials) and neuronal activity.
# - Use electrode locations/mappings for spatial analysis.
# - Examine more assets or other sessions within the Dandiset.
#
# ---
#
# *Notebook generated via AI. Please verify all steps and consult raw Dandiset resources for scientific research.*