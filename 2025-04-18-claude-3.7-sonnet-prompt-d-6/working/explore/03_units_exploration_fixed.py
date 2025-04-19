#!/usr/bin/env python
"""
This script explores the spike times data for units in the NWB file.
We'll analyze spike rates, inter-spike intervals, and spiking patterns.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the units table
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")
print(f"Unit IDs: {units_df.index.tolist()}")

# Get trial information
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
recording_duration = nwb.acquisition['time_series'].data.shape[0] / nwb.acquisition['time_series'].rate
print(f"Recording duration: {recording_duration:.2f} seconds")

# Calculate basic statistics for each unit
unit_stats = []

# Get the actual unit IDs from the DataFrame
unit_ids = units_df.index.tolist()

for i, unit_id in enumerate(unit_ids):
    # Get spike times for this unit
    spike_times = nwb.units['spike_times'][i]  # Use the index i instead of unit_id
    
    # Calculate statistics
    n_spikes = len(spike_times)
    mean_rate = n_spikes / recording_duration
    
    if n_spikes > 1:
        # Calculate inter-spike intervals
        isis = np.diff(spike_times)
        mean_isi = np.mean(isis)
        cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else np.nan
    else:
        mean_isi = np.nan
        cv_isi = np.nan
    
    unit_stats.append({
        'unit_id': unit_id,
        'index': i,
        'n_spikes': n_spikes,
        'mean_rate': mean_rate,
        'mean_isi': mean_isi,
        'cv_isi': cv_isi
    })

# Convert to DataFrame
unit_stats_df = pd.DataFrame(unit_stats)
print("\nUnit statistics summary:")
print(f"Average spike rate across units: {unit_stats_df['mean_rate'].mean():.2f} Hz")
print(f"Min spike rate: {unit_stats_df['mean_rate'].min():.2f} Hz")
print(f"Max spike rate: {unit_stats_df['mean_rate'].max():.2f} Hz")

# Plot spike rate distribution
plt.figure(figsize=(10, 6))
plt.hist(unit_stats_df['mean_rate'], bins=20)
plt.xlabel('Mean Spike Rate (Hz)')
plt.ylabel('Number of Units')
plt.title('Distribution of Mean Spike Rates Across Units')
plt.savefig('explore/spike_rate_dist.png')

# Plot inter-spike interval coefficient of variation
plt.figure(figsize=(10, 6))
valid_mask = ~np.isnan(unit_stats_df['cv_isi'])
plt.scatter(
    unit_stats_df.loc[valid_mask, 'mean_rate'], 
    unit_stats_df.loc[valid_mask, 'cv_isi']
)
plt.xlabel('Mean Spike Rate (Hz)')
plt.ylabel('Coefficient of Variation of ISI')
plt.title('Regularity of Firing vs. Mean Spike Rate')
plt.savefig('explore/cv_isi_vs_rate.png')

# Select a few units to examine in detail (by index)
top_units = unit_stats_df.sort_values('mean_rate', ascending=False).head(5)
selected_indices = top_units['index'].values
selected_unit_ids = top_units['unit_id'].values

# Create raster plots for these units
plt.figure(figsize=(12, 8))
for i, (idx, unit_id) in enumerate(zip(selected_indices, selected_unit_ids)):
    spike_times = nwb.units['spike_times'][idx]
    plt.scatter(spike_times, np.ones_like(spike_times) * i, marker='|', s=20)

plt.yticks(range(len(selected_indices)), [f"Unit {u}" for u in selected_unit_ids])
plt.xlabel('Time (s)')
plt.title('Spike Raster Plot for Top 5 Units by Firing Rate')
plt.xlim(0, 300)  # Look at first 5 minutes
plt.savefig('explore/spike_raster.png')

# Examine spiking around trials
# Select a couple of trials
selected_trials = trials_df.iloc[:5]  
window = 2.0  # seconds before and after trial start

plt.figure(figsize=(15, 10))
for unit_idx, (idx, unit_id) in enumerate(zip(selected_indices[:3], selected_unit_ids[:3])):  # Look at top 3 units
    spike_times = nwb.units['spike_times'][idx]
    
    for trial_idx, (trial_id, trial) in enumerate(selected_trials.iterrows()):
        trial_start = trial['start_time']
        trial_end = trial['stop_time']
        
        # Find spikes within window around trial start
        mask = (spike_times >= trial_start - window) & (spike_times <= trial_end + window)
        trial_spikes = spike_times[mask]
        
        # Normalize times relative to trial start
        rel_times = trial_spikes - trial_start
        
        # Plot
        row_idx = unit_idx * len(selected_trials) + trial_idx
        plt.scatter(rel_times, np.ones_like(rel_times) * row_idx, marker='|', s=30)
        
        # Mark trial boundaries
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=trial_end - trial_start, color='g', linestyle='--', alpha=0.5)

# Create labels
yticks = []
yticklabels = []
for unit_idx, unit_id in enumerate(selected_unit_ids[:3]):
    for trial_idx, (trial_id, _) in enumerate(selected_trials.iterrows()):
        row_idx = unit_idx * len(selected_trials) + trial_idx
        yticks.append(row_idx)
        yticklabels.append(f"U{unit_id}-T{trial_id}")

plt.yticks(yticks, yticklabels)
plt.xlabel('Time Relative to Trial Start (s)')
plt.title('Spike Activity Around Trial Boundaries')
plt.grid(axis='x', alpha=0.3)
plt.savefig('explore/trial_aligned_spikes.png')

# Close the file handles
h5_file.close()
remote_file.close()