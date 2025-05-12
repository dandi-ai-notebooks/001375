# %% [markdown]
# # Exploring Dandiset 001375: Septum GABA disruption with DREADDs

# %% [markdown]
# > **Note**: This notebook was AI-generated and has not been fully verified. Please exercise caution when interpreting the code or results. Always verify important findings with your own analysis.

# %% [markdown]
# ## Overview
# 
# This notebook explores [Dandiset 001375 (version 0.250406.1855)](https://dandiarchive.org/dandiset/001375/0.250406.1855), which contains electrophysiological recordings from mice with DREADDs manipulations targeting GABAergic interneurons in the medial septum.
# 
# ### Experimental Background
# 
# **DREADD Technology**: Designer Receptors Exclusively Activated by Designer Drugs (DREADDs) are a chemogenetic tool that allows researchers to selectively manipulate specific neuronal populations. In this study, DREADDs were used to suppress activity of GABAergic interneurons in the medial septum.
# 
# **Significance of Septal GABAergic Neurons**: The medial septum contains GABAergic neurons that project to the hippocampus and are crucial for generating theta oscillations, which are important for spatial navigation and memory. By suppressing these neurons, researchers can examine their role in hippocampal and neocortical activity.
# 
# **Expected Effects**: Disruption of septal GABAergic activity would be expected to:
# - Reduce hippocampal theta oscillations
# - Potentially alter firing patterns of hippocampal neurons
# - Affect coordination between hippocampus and neocortex
# - Possibly influence spatial navigation performance
# 
# The dataset contains recordings from mice running laps in a virtual hallway, allowing the investigation of neural activity in the ventral hippocampus and visual cortex in relation to the animal's behavior.
# 
# ### Dandiset Description
# 
# - **Title**: Septum GABA disruption with DREADDs
# - **Description**: Pilot study of the effect of disrupting septal gabaergic activity using DREADDs on hippocampal and neocortical activity
# - **Contributors**: Eckert, Michael; NIH Brain; McNaughton, Bruce; Ferbinteanu, Janina
# 
# This notebook will:
# 1. Load and explore the structure of the Dandiset
# 2. Examine the metadata and organization of NWB files
# 3. Explore trial structure and behavioral data
# 4. Analyze neural activity and spike patterns
# 5. Visualize raw electrophysiology data
# 6. Investigate relationships between neural activity and behavior

# %% [markdown]
# ## Required Packages
# 
# The following packages are required to run this notebook:

# %%
import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import signal
from itertools import islice

# Set seaborn styling for plots
sns.set_theme()

# %% [markdown]
# ## Computational Considerations for Large NWB Files
# 
# Working with NWB files, especially large ones like those in this dataset (50-70GB each), requires careful consideration of computational resources and data handling strategies. Here are some key points to keep in mind:
# 
# ### Remote Access vs. Local Storage
# 
# In this notebook, we use `remfile.File` to stream data directly from the DANDI archive without downloading the entire file. This approach has advantages and limitations:
# 
# **Advantages:**
# - No need to download massive files
# - Can start analysis immediately
# - Conserves local disk space
# 
# **Limitations:**
# - Slower access times compared to local files
# - Dependent on network connectivity
# - Continuous network transfer during analysis
# 
# ### Accessing Data Efficiently
# 
# When working with large NWB files, it's important to:
# 
# 1. **Access only the data you need**: Instead of loading entire datasets, extract only the required portions
# 2. **Use indexing efficiently**: For example, instead of `time_series.data[:]`, use `time_series.data[start_idx:end_idx, :]`
# 3. **Process data in chunks**: When analyzing long recordings, process the data in manageable segments
# 
# ### Example: Loading a Specific Time Window
# 
# Here's how to efficiently load a specific time window of data:

# %%
# Function to load a specific time window of data
def load_time_window(time_series, start_time, duration, channels=None):
    """
    Load a specific time window of data from a time series.
    
    Parameters:
    -----------
    time_series : TimeSeries
        The time series object.
    start_time : float
        Start time in seconds.
    duration : float
        Duration of the window in seconds.
    channels : list or None
        List of channel indices to load. If None, load all channels.
    
    Returns:
    --------
    data : ndarray
        Data array for the specified time window and channels.
    t : ndarray
        Time points for the loaded data.
    """
    # Convert times to indices
    start_idx = int(start_time * time_series.rate)
    end_idx = int((start_time + duration) * time_series.rate)
    
    # Load data
    if channels is not None:
        data = time_series.data[start_idx:end_idx, channels]
    else:
        data = time_series.data[start_idx:end_idx, :]
    
    # Generate time points
    t = np.linspace(start_time, start_time + duration, end_idx - start_idx)
    
    return data, t

# Example usage (not executed to avoid unnecessary data loading)
print("Example function for loading a 5-second window of data starting at 10s from 5 channels")
print("Usage: data, t = load_time_window(time_series, start_time=10, duration=5, channels=[0, 1, 2, 3, 4])")

# %% [markdown]
# ## Loading the Dandiset
# 
# We'll use the DANDI API to connect to the archive and access information about the Dandiset.

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001375", "0.250406.1855")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Description: {metadata['description']}")

# Handle the contributor field which may contain dictionaries
contributors = metadata['contributor']
if contributors and isinstance(contributors[0], dict):
    # If contributors are dictionaries, extract relevant information
    contributor_names = [c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in contributors]
    print(f"Contributors: {', '.join(contributor_names)}")
else:
    # If contributors are strings
    print(f"Contributors: {', '.join(contributors)}")

# %% [markdown]
# ## Exploring Dandiset Assets
# 
# Let's list the assets (NWB files) contained in this Dandiset.

# %%
# Get and display the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"Number of assets: {len(assets)}")

print("\nAsset details:")
for i, asset in enumerate(assets):
    print(f"{i+1}. {asset.path} (Size: {asset.size / 1e9:.2f} GB)")

# %% [markdown]
# The Dandiset contains three NWB files from two subjects:
# 
# 1. `sub-MS13B/sub-MS13B_ses-20240725T190000_ecephys.nwb`
# 2. `sub-MS14A/sub-MS14A_ses-20240906T190000_obj-12781w8_ecephys.nwb` 
# 3. `sub-MS14A/sub-MS14A_ses-20240906T190000_obj-ardad2_ecephys.nwb`
# 
# For this analysis, we'll focus on the first file (sub-MS13B).

# %% [markdown]
# ## Loading an NWB File
# 
# Let's load the first NWB file and examine its structure. Note that NWB files in this dataset are quite large (50-70GB), so loading may take some time. 
# 
# > **Important**: The NWB files are streamed remotely using the remfile package, which allows us to access parts of the data without downloading the entire file. However, accessing the data may still be slow due to the file size and network bandwidth.

# %%
# Select the first asset
asset = assets[0]
print(f"Selected asset: {asset.path}")

# Get the URL for the asset
asset_url = asset.get_content_url(follow_redirects=1)
print(f"Asset URL: {asset_url}")

# Load the NWB file
remote_file = remfile.File(asset_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## Examining NWB File Metadata
# 
# Let's examine the basic metadata in the NWB file.

# %%
# Print basic information
print(f"Identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"File creation date: {nwb.file_create_date[0]}")
print(f"Timestamps reference time: {nwb.timestamps_reference_time}")

# Subject information
print("\nSubject Information:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# %% [markdown]
# ## Exploring the Data Structure
# 
# Let's explore what kinds of data are available in the NWB file.

# %%
# Acquisition data
print("Acquisition data keys:")
for key in nwb.acquisition:
    print(f"- {key}")

# Device information
print("\nDevices:")
for device_name, device in nwb.devices.items():
    print(f"- {device_name}: {device.description} (Manufacturer: {device.manufacturer})")

# Electrode groups
print("\nElectrode Groups:")
for group_name, group in nwb.electrode_groups.items():
    print(f"- {group_name}: {group.description}, located in {group.location}")

# %% [markdown]
# ## Exploring Electrode Information
# 
# The dataset contains information about electrodes used for recording. Let's examine this information.

# %%
# Electrode information
num_electrodes = len(nwb.electrodes)
print(f"Number of electrodes: {num_electrodes}")
print(f"Electrode columns: {nwb.electrodes.colnames}")

# Get electrode dataframe
electrode_df = nwb.electrodes.to_dataframe()
print("\nElectrode dataframe sample (first 5 rows):")
print(electrode_df.head())

# Count electrodes by group
print("\nElectrodes by group:")
group_counts = electrode_df['group_name'].value_counts()
for group_name, count in group_counts.items():
    print(f"- {group_name}: {count} electrodes")

# %% [markdown]
# ## Exploring Raw Electrophysiology Data
# 
# Let's examine the raw electrophysiology data in the NWB file.

# %%
# Get information about the raw data
time_series = nwb.acquisition["time_series"]
print(f"Raw data dimensions: {time_series.data.shape}")
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Units: {time_series.unit}")
print(f"Data type: {time_series.data.dtype}")

# %% [markdown]
# Each row in the raw data represents a time point, and each column represents an electrode. The data are sampled at 30,000 Hz, providing high-resolution recordings of neural activity.

# %% [markdown]
# ## Visualizing Raw Electrophysiology Data
# 
# Let's visualize a small segment (1 second) of raw data from a sample of electrodes.

# %%
# Select a small time window to analyze (1 second of data starting at 10 seconds)
start_time = 10  # seconds
duration = 1.0   # seconds
start_idx = int(start_time * time_series.rate)
end_idx = int((start_time + duration) * time_series.rate)

# Select a subset of electrodes (3 from each shank)
shank1_electrodes = electrode_df[electrode_df['group_name'] == 'shank1'].head(3).index
shank2_electrodes = electrode_df[electrode_df['group_name'] == 'shank2'].head(3).index
selected_electrodes = list(shank1_electrodes) + list(shank2_electrodes)
selected_labels = [electrode_df.loc[idx, 'label'] for idx in selected_electrodes]

print(f"Selected time window: {start_time}s to {start_time + duration}s")
print(f"Selected electrodes: {selected_labels}")

# Extract the data for the selected time window and electrodes
data_segment = time_series.data[start_idx:end_idx, selected_electrodes]

# Plot the raw data for the selected electrodes
plt.figure(figsize=(12, 8))
t = np.linspace(start_time, start_time + duration, end_idx - start_idx)

for i, (electrode_idx, label) in enumerate(zip(selected_electrodes, selected_labels)):
    # Offset each trace for visibility
    offset = i * 500
    
    # Plot with appropriate offset
    plt.plot(t, data_segment[:, i] + offset, linewidth=0.8, 
             label=f"{label} (#{electrode_idx})")

plt.xlabel('Time (s)')
plt.ylabel('Signal (μV) + Offset')
plt.title('Raw Electrophysiology Data for Selected Electrodes')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# The plot above shows the raw electrophysiology data for 6 selected electrodes (3 from each shank) over a 1-second period. We can observe:
# 
# 1. Clear neural activity with visible spike events
# 2. Synchronized activity across electrodes, suggesting coherent neural events
# 3. Differences in signal amplitude and baseline between electrodes
# 
# These raw signals contain both local field potentials (the slower oscillations) and multi-unit activity (the fast spikes).

# %% [markdown]
# ## Examining Trial Information
# 
# The NWB file contains trial information, which marks the start and stop times of behavioral trials (laps in the virtual hallway).

# %%
# Get trial information
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
print(f"Trial columns: {nwb.trials.colnames}")

# Show first 5 trials
print("\nFirst 5 trials:")
print(trials_df.head())

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
print(f"\nMean trial duration: {np.mean(trials_df['duration']):.3f} s")
print(f"Median trial duration: {np.median(trials_df['duration']):.3f} s")
print(f"Min trial duration: {np.min(trials_df['duration']):.3f} s")
print(f"Max trial duration: {np.max(trials_df['duration']):.3f} s")

# %% [markdown]
# Let's visualize the trial durations to understand the behavioral structure of the experiment.

# %%
# Plot trial durations
plt.figure(figsize=(12, 6))
plt.plot(range(len(trials_df)), trials_df['duration'], 'o-', markersize=3)
plt.xlabel('Trial Number')
plt.ylabel('Duration (s)')
plt.title('Trial Durations')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# The plot shows the duration of each trial (lap) in the experiment. Most trials last between 5-15 seconds, but there are occasional longer trials. One trial (#215) is particularly long, lasting over 160 seconds, which could represent an anomaly or a period where the animal was not actively engaged in the task.
#
# The consistency of trial durations suggests the animal was well-trained on the task and performed it with relatively stable speed throughout the session.

# %% [markdown]
# Let's also look at the distribution of trial durations.

# %%
# Plot the distribution of trial durations
plt.figure(figsize=(10, 5))
plt.hist(trials_df['duration'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Trial Duration (s)')
plt.ylabel('Count')
plt.title('Distribution of Trial Durations')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# The histogram clearly shows that most trials lasted between 7-15 seconds, with a small number of longer outliers.

# %% [markdown]
# ## Exploring Neural Units (Sorted Spikes)
# 
# The NWB file contains information about sorted spike units, which represent the activity of individual neurons or small groups of neurons.

# %%
# Units information
print(f"Number of units: {len(nwb.units)}")
print(f"Unit columns: {nwb.units.colnames}")
print(f"Waveform unit: {nwb.units.waveform_unit}")

# Get unit IDs
unit_ids = nwb.units.id.data[:]
print(f"Unit IDs: {unit_ids[:10]}...")  # Show first 10 unit IDs

# Show spike counts for first 5 units
print("\nFirst 5 units spike counts:")
for i in range(min(5, len(nwb.units))):
    spike_times = nwb.units["spike_times"][i]
    print(f"Unit {unit_ids[i]}: {len(spike_times)} spikes")

# %% [markdown]
# ## Visualizing Spike Activity
# 
# Let's visualize the spike activity of a few units to understand their firing patterns.

# %%
# Select 3 units with different spike counts for visualization
selected_units = [0, 1, 2]  # First three units
print(f"Selected units: {[unit_ids[i] for i in selected_units]}")

# Get spike times for selected units
spike_times_list = []
for unit_idx in selected_units:
    spike_times = nwb.units["spike_times"][unit_idx]
    spike_times_list.append(spike_times)
    print(f"Unit {unit_ids[unit_idx]}: {len(spike_times)} spikes")

# %% [markdown]
# ### Spike Raster Plot
# 
# Let's create a raster plot showing spike times for the selected units during the first 100 seconds of the recording.

# %%
# Create a raster plot for selected units during first 100 seconds
plt.figure(figsize=(15, 5))
for i, (unit_idx, spike_times) in enumerate(zip(selected_units, spike_times_list)):
    # Filter spike times to first 100 seconds
    spikes_in_window = spike_times[spike_times < 100]
    plt.plot(spikes_in_window, np.ones_like(spikes_in_window) * i + 1, '|', markersize=4, 
             label=f"Unit {unit_ids[unit_idx]}")

plt.yticks(np.arange(1, len(selected_units) + 1), [f"Unit {unit_ids[i]}" for i in selected_units])
plt.xlabel('Time (s)')
plt.title('Spike Raster Plot (First 100 seconds)')
plt.xlim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# The raster plot shows the firing patterns of three different units. We can observe:
# 
# - Unit 1 has sparse, irregular firing with occasional clusters of activity
# - Unit 2 shows a moderately high, consistent firing rate throughout the time period
# - Unit 3 has an extremely high firing rate, appearing almost continuous
# 
# These different firing patterns likely reflect different types of neurons or functional roles within the brain regions being recorded.

# %% [markdown]
# ### Firing Rate Analysis
# 
# Let's analyze the firing rates of these units across trials to see if there are patterns related to the behavior.

# %%
# Calculate firing rates for each trial and unit
trial_firing_rates = np.zeros((len(trials_df), len(selected_units)))

for t, (_, trial) in enumerate(trials_df.iterrows()):
    start, stop = trial['start_time'], trial['stop_time']
    duration = stop - start
    
    for u, (unit_idx, spike_times) in enumerate(zip(selected_units, spike_times_list)):
        # Count spikes in this trial
        trial_spikes = np.sum((spike_times >= start) & (spike_times < stop))
        rate = trial_spikes / duration
        trial_firing_rates[t, u] = rate

# Plot firing rates across trials
plt.figure(figsize=(15, 10))

for u, unit_idx in enumerate(selected_units):
    plt.subplot(len(selected_units), 1, u+1)
    plt.plot(range(len(trials_df)), trial_firing_rates[:, u], 'o-', markersize=2)
    plt.axhline(y=np.median(trial_firing_rates[:, u]), color='r', linestyle='--', alpha=0.5, 
                label=f'Median: {np.median(trial_firing_rates[:, u]):.1f} Hz')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_ids[unit_idx]} Firing Rate Across Trials')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.xlabel('Trial Number')
plt.tight_layout()

# %% [markdown]
# These plots show how the firing rates of each unit changed across trials. Some interesting observations:
# 
# 1. Unit 1 has a relatively low firing rate (around 4-5 Hz) with some variations across trials
# 2. Unit 2 shows a moderate firing rate (around 20-25 Hz) with a slight downward trend in the later trials
# 3. Unit 3 exhibits the most dramatic change, with a clear downward adaptation from around 80 Hz in early trials to around 40-50 Hz by trial 250, followed by a slight recovery
# 
# These patterns could reflect neural adaptation, learning, or changes in the animal's state or behavior throughout the experiment.

# %% [markdown]
# ### Relationship Between Neural Activity and Behavior
# 
# Let's examine the relationship between neural activity and behavior in several ways:
# 
# 1. Correlation between trial duration and firing rates
# 2. Neural activity aligned with trial events

# %%
# Check for correlation between trial duration and firing rates
plt.figure(figsize=(15, 5))
for u, unit_idx in enumerate(selected_units):
    plt.subplot(1, len(selected_units), u+1)
    
    # Calculate correlation
    r, p = stats.pearsonr(trials_df['duration'], trial_firing_rates[:, u])
    
    plt.scatter(trials_df['duration'], trial_firing_rates[:, u], alpha=0.5)
    plt.xlabel('Trial Duration (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_ids[unit_idx]}: r={r:.3f}, p={p:.3f}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()

# %% [markdown]
# ### Neural Activity Aligned to Trial Events
# 
# To better understand how neural activity relates to behavior, let's visualize spike activity aligned with the start and end of trials. This can reveal patterns of activity that are specifically associated with certain behavioral events.

# %%
# Create a function to align spikes to trial events
def align_spikes_to_events(spike_times, event_times, window=(-2, 2)):
    """
    Aligns spike times to event times within a specified window.
    
    Parameters:
    -----------
    spike_times : array-like
        Array of spike times.
    event_times : array-like
        Array of event times to align to.
    window : tuple
        Time window around each event (in seconds).
        
    Returns:
    --------
    aligned_spikes : list of arrays
        List of arrays containing spike times aligned to each event.
    """
    aligned_spikes = []
    
    for event in event_times:
        # Find spikes within window around the event
        mask = (spike_times >= event + window[0]) & (spike_times <= event + window[1])
        # Align to event time (t=0)
        aligned_spikes.append(spike_times[mask] - event)
    
    return aligned_spikes

# Select a subset of trials for clarity
num_trials = 50
trial_subset = trials_df.iloc[:num_trials]

# Align spikes to trial starts and ends
trial_starts = trial_subset['start_time'].values
trial_ends = trial_subset['stop_time'].values

# List to store all aligned spikes for plotting
aligned_start = []
aligned_end = []

# Time window around events (in seconds)
window = (-1, 1)

# Align spikes for each unit
for unit_idx, spike_times in zip(selected_units, spike_times_list):
    aligned_start.append(align_spikes_to_events(spike_times, trial_starts, window))
    aligned_end.append(align_spikes_to_events(spike_times, trial_ends, window))

# Plot aligned spike rasters
plt.figure(figsize=(15, 10))

# Plot spikes aligned to trial starts
for u, unit_idx in enumerate(selected_units):
    plt.subplot(len(selected_units), 2, 2*u+1)
    
    # Plot each trial
    for trial, spikes in enumerate(aligned_start[u][:num_trials]):
        plt.plot(spikes, np.ones_like(spikes) * trial, '|', color=f'C{u}', markersize=3)
    
    plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
    plt.xlim(window)
    plt.xlabel('Time from Trial Start (s)')
    plt.ylabel('Trial #')
    plt.title(f'Unit {unit_ids[unit_idx]} - Aligned to Trial Start')
    if u == 0:
        plt.legend()

# Plot spikes aligned to trial ends
for u, unit_idx in enumerate(selected_units):
    plt.subplot(len(selected_units), 2, 2*u+2)
    
    # Plot each trial
    for trial, spikes in enumerate(aligned_end[u][:num_trials]):
        plt.plot(spikes, np.ones_like(spikes) * trial, '|', color=f'C{u}', markersize=3)
    
    plt.axvline(x=0, color='r', linestyle='--', label='Trial End')
    plt.xlim(window)
    plt.xlabel('Time from Trial End (s)')
    plt.ylabel('Trial #')
    plt.title(f'Unit {unit_ids[unit_idx]} - Aligned to Trial End')
    if u == 0:
        plt.legend()

plt.tight_layout()

# %%
# Calculate peristimulus time histograms (PSTH) for trial starts and ends
bin_size = 0.05  # 50 ms bins
bins = np.arange(window[0], window[1]+bin_size, bin_size)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Calculate PSTHs
psth_start = np.zeros((len(selected_units), len(bins)-1))
psth_end = np.zeros((len(selected_units), len(bins)-1))

for u in range(len(selected_units)):
    # Combine spikes from all trials
    all_start_spikes = np.concatenate(aligned_start[u])
    all_end_spikes = np.concatenate(aligned_end[u])
    
    # Calculate histograms
    psth_start[u], _ = np.histogram(all_start_spikes, bins=bins)
    psth_end[u], _ = np.histogram(all_end_spikes, bins=bins)
    
    # Convert to firing rate (Hz)
    psth_start[u] = psth_start[u] / (bin_size * len(trial_subset))
    psth_end[u] = psth_end[u] / (bin_size * len(trial_subset))

# Plot PSTHs
plt.figure(figsize=(15, 10))

for u, unit_idx in enumerate(selected_units):
    plt.subplot(len(selected_units), 2, 2*u+1)
    plt.bar(bin_centers, psth_start[u], width=bin_size*0.8, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
    plt.xlabel('Time from Trial Start (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_ids[unit_idx]} - PSTH Aligned to Trial Start')
    if u == 0:
        plt.legend()
    
    plt.subplot(len(selected_units), 2, 2*u+2)
    plt.bar(bin_centers, psth_end[u], width=bin_size*0.8, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Trial End')
    plt.xlabel('Time from Trial End (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_ids[unit_idx]} - PSTH Aligned to Trial End')
    if u == 0:
        plt.legend()

plt.tight_layout()

# %% [markdown]
# The scatter plots show the relationship between trial duration and firing rates for each unit:
# 
# 1. Unit 1 shows no significant correlation (r=-0.039, p=0.444)
# 2. Unit 2 shows a significant negative correlation (r=-0.170, p=0.001), with lower firing rates during longer trials
# 3. Unit 3 also shows a significant negative correlation (r=-0.122, p=0.017), similar to Unit 2
# 
# The negative correlations for Units 2 and 3 suggest these neurons may be less active during longer trials, or that their activity might facilitate faster task completion.

# %% [markdown]
# ## Frequency Analysis of Raw Neural Signals
# 
# Let's analyze the frequency content of the raw neural signals to understand the oscillatory patterns present in the data. Neural oscillations occur in distinct frequency bands that have been associated with different cognitive and behavioral states:
# 
# - **Delta (0.5-4 Hz)**: Associated with deep sleep
# - **Theta (4-8 Hz)**: Important for spatial navigation and memory, particularly in the hippocampus
# - **Alpha (8-12 Hz)**: Often associated with wakeful rest
# - **Beta (12-30 Hz)**: Associated with normal waking consciousness and motor control
# - **Gamma (30-100 Hz)**: Associated with cognitive processing and coordinated neural activity
# 
# Given that the experiment involves disrupting septal GABAergic neurons, which are crucial for generating theta oscillations, we should pay particular attention to power in the theta band.

# %%
# Select a 1-second segment of data for frequency analysis
segment_length = int(1.0 * time_series.rate)  # 1 second at 30kHz
data_segment = time_series.data[start_idx:start_idx + segment_length, selected_electrodes[:6]]

# Define frequency bands of interest
freq_bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 100)
}

# Import scipy.signal for spectrum analysis
from scipy import signal as sg

# Compute power spectrum for each electrode
plt.figure(figsize=(14, 8))

# First plot: Full spectrum (0-500 Hz)
plt.subplot(2, 1, 1)
for i, (electrode_idx, label) in enumerate(zip(selected_electrodes[:6], selected_labels[:6])):
    signal_data = data_segment[:, i]
    
    # Compute power spectrum using scipy's welch
    freq, ps = sg.welch(signal_data, fs=time_series.rate, nperseg=min(8192, len(signal_data)))
    
    # Plot only positive frequencies up to 500Hz for clarity
    mask = (freq > 0) & (freq <= 500)
    plt.semilogy(freq[mask], ps[mask], linewidth=0.8, alpha=0.7,
                label=f"{label} (#{electrode_idx})")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectrum (0-500 Hz)')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)

# Second plot: Zoomed in to lower frequencies (0-50 Hz)
plt.subplot(2, 1, 2)
for i, (electrode_idx, label) in enumerate(zip(selected_electrodes[:6], selected_labels[:6])):
    signal_data = data_segment[:, i]
    
    # Compute power spectrum using scipy's welch
    freq, ps = sg.welch(signal_data, fs=time_series.rate, nperseg=min(8192, len(signal_data)))
    
    # Plot only frequencies in range of interest
    mask = (freq > 0) & (freq <= 50)
    plt.semilogy(freq[mask], ps[mask], linewidth=1.0, alpha=0.7,
                label=f"{label} (#{electrode_idx})")

# Add colored bands for different frequency ranges
y_min, y_max = plt.ylim()
for name, (fmin, fmax) in freq_bands.items():
    if fmax <= 50:  # Only show bands up to 50 Hz
        plt.fill_betweenx([y_min, y_max], fmin, fmax, color=f'C{list(freq_bands.keys()).index(name)}', alpha=0.1)
        plt.text(fmin + (fmax-fmin)/2, y_max*0.5, name, 
                ha='center', va='bottom', rotation=90, fontsize=8)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectrum (0-50 Hz) with Physiological Frequency Bands')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# The power spectrum shows the frequency content of the raw neural signals:
# 
# 1. The highest power is in the low frequency range (below 50 Hz), which includes delta (0-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), and beta (13-30 Hz) oscillations
# 2. There's a clear 1/f pattern (power decreases with increasing frequency), which is typical for neural signals
# 3. There are differences between electrodes, with shank1 electrodes generally showing higher power across frequencies
# 4. The absence of pronounced 60 Hz peaks suggests minimal power line interference
# 
# These oscillatory patterns reflect coordinated network activity in the recorded brain regions.

# %% [markdown]
# ## Correlation Analysis Between Electrodes
# 
# Let's examine the correlation between signals from different electrodes to understand the spatial organization of neural activity.

# %%
# Create a pairwise correlation matrix for selected electrodes
plt.figure(figsize=(10, 8))

# Calculate correlation matrix
corr_matrix = np.corrcoef(data_segment.T)

# Plot as heatmap
sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, 
            xticklabels=selected_labels[:6], yticklabels=selected_labels[:6])
plt.title('Signal Correlation Between Electrodes')
plt.tight_layout()

# %% [markdown]
# The correlation matrix reveals spatial patterns in neural activity:
# 
# 1. Strong positive correlations among electrodes within the same shank (e.g., all shank1 electrodes correlate strongly with each other)
# 2. Weaker correlations between electrodes from different shanks
# 3. Gradual decrease in correlation as electrode distance increases
# 
# This suggests that the two shanks are recording from different neural circuits or populations, and that neural activity is spatially organized with signal similarity decreasing with distance.

# %% [markdown]
# ## Summary and Conclusions
# 
# In this notebook, we explored Dandiset 001375 (version 0.250406.1855), which contains electrophysiological recordings from mice with DREADDs manipulations targeting GABAergic interneurons in the medial septum. Here's what we found:
# 
# 1. **Dataset Structure**: The Dandiset contains three NWB files from two subjects, with recordings from mice running laps in a virtual hallway.
# 
# 2. **Recording Setup**: The recordings were made using silicon probes with two shanks, each with 128 electrodes, positioned in the ventral hippocampus and visual cortex.
# 
# 3. **Behavioral Structure**: The experiment consisted of ~387 trials (laps) with typical durations of 5-15 seconds, showing consistent behavioral performance throughout the session.
# 
# 4. **Neural Activity**:
#    - We observed different types of neural units with distinct firing patterns
#    - Some units showed adaptation across trials, with decreasing firing rates over time
#    - Units 2 and 3 displayed negative correlations between firing rate and trial duration
#    - Spike activity showed patterns associated with trial starts and ends, suggesting task-related neural responses
# 
# 5. **Signal Characteristics**:
#    - Raw signals contained both local field potentials and multi-unit activity
#    - Power spectrum analysis revealed prominent low-frequency oscillations, including in the theta band
#    - Electrodes within the same shank showed stronger signal correlations
# 
# ### Interpretation in the Context of DREADDs Manipulation
# 
# The DREADD manipulation targeting GABAergic interneurons in the medial septum was designed to suppress the activity of these neurons, which are known to project to the hippocampus and regulate theta oscillations. Our exploration revealed several findings that may relate to this manipulation:
# 
# 1. **Neural Adaptation**: The pronounced adaptation of Unit 3 over the course of the experiment could reflect dynamic changes in network activity following DREADD-mediated suppression of GABAergic input.
# 
# 2. **Correlations with Behavior**: The negative correlations between firing rates and trial durations for Units 2 and 3 may indicate that disruption of septal GABAergic signaling affects the relationship between neural activity and behavioral performance.
# 
# 3. **Oscillatory Activity**: The power spectrum analysis showed prominent low-frequency activity, but a focused analysis comparing theta power between control and manipulation conditions would be needed to fully assess the impact of the DREADD manipulation.
# 
# 4. **Trial-Related Neural Responses**: The presence of trial-aligned neural activity suggests that despite potential disruption of septal input, hippocampal and cortical neurons still maintain task-related firing patterns.
# 
# These findings provide preliminary insights into how GABAergic activity in the medial septum might influence neural activity in the hippocampus and visual cortex during spatial navigation tasks. However, a more comprehensive analysis comparing control and manipulation conditions would be necessary to draw definitive conclusions.

# %% [markdown]
# ## Future Directions
# 
# Based on this exploration, several directions for further analysis could be interesting:
# 
# 1. **Specific Manipulation Effects**: Comparing the neural activity between control and DREADD manipulation conditions to directly assess the effect of disrupting septal GABAergic activity
# 
# 2. **Oscillatory Changes**: Conducting a detailed analysis of changes in neural oscillations, particularly theta (4-8 Hz) and gamma (30-100 Hz) power, which are known to be regulated by septal inputs
# 
# 3. **Network Coordination**: Investigating the coordination between hippocampus and visual cortex during spatial navigation, which might be disrupted by the manipulation
# 
# 4. **Spatial Coding Analysis**: Examining place cell activity and spatial information content to determine if disruption of septal GABAergic input affects spatial representation
# 
# 5. **Computational Modeling**: Developing computational models to simulate the effects of reduced GABAergic input on hippocampal network activity
# 
# 6. **Cross-Subject Comparison**: Analyzing differences in neural activity between the two subjects to assess individual variability in response to the manipulation
# 
# 7. **Ripple Analysis**: Examining sharp-wave ripples in the hippocampal recordings, as these high-frequency events might be affected by changes in GABAergic input
# 
# These analyses would provide deeper insights into how septal GABAergic neurons influence hippocampal and neocortical activity during spatial navigation and could contribute to our understanding of the role of this circuit in spatial memory and cognition.

# %% [markdown]
# ## Accessing the NWB File on Neurosift
# 
# You can explore the NWB file interactively on Neurosift using the following link:
# 
# [View on Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/&dandisetId=001375&dandisetVersion=0.250406.1855)