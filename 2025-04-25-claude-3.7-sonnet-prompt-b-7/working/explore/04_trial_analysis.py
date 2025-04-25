"""
This script explores the trial structure in the first NWB file and examines 
the relationship between neural activity (from units) and trials.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import seaborn as sns

# Set up the plotting style
sns.set_theme()
plt.rcParams['figure.figsize'] = (14, 8)

# Create a directory for images if it doesn't exist
if not os.path.exists('explore/images'):
    os.makedirs('explore/images')

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial information
print("Analyzing trial structure...")
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
print(f"First 5 trials:")
print(trials_df.head())

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']

# Analyze trial durations
print("\nTrial duration statistics:")
print(f"Mean duration: {trials_df['duration'].mean():.2f} seconds")
print(f"Median duration: {trials_df['duration'].median():.2f} seconds")
print(f"Min duration: {trials_df['duration'].min():.2f} seconds")
print(f"Max duration: {trials_df['duration'].max():.2f} seconds")
print(f"Std dev of duration: {trials_df['duration'].std():.2f} seconds")

# Plot trial durations
plt.figure(figsize=(14, 6))
plt.plot(trials_df.index, trials_df['duration'], 'o-', alpha=0.6)
plt.xlabel('Trial Number')
plt.ylabel('Duration (seconds)')
plt.title('Trial Durations')
plt.grid(True, alpha=0.3)
plt.savefig('explore/images/trial_durations.png')

# Plot histogram of trial durations
plt.figure(figsize=(14, 6))
plt.hist(trials_df['duration'], bins=30, alpha=0.7)
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.title('Distribution of Trial Durations')
plt.grid(True, alpha=0.3)
plt.savefig('explore/images/trial_duration_histogram.png')

# Plot trial start and stop times to see the timing structure
plt.figure(figsize=(14, 6))
plt.scatter(trials_df.index, trials_df['start_time'], label='Start Time', alpha=0.6)
plt.scatter(trials_df.index, trials_df['stop_time'], label='Stop Time', alpha=0.6)
plt.xlabel('Trial Number')
plt.ylabel('Time (seconds)')
plt.title('Trial Start and Stop Times')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('explore/images/trial_start_stop_times.png')

# Calculate inter-trial intervals
trials_df['inter_trial_interval'] = trials_df['start_time'].diff()

print("\nInter-trial interval statistics:")
print(f"Mean ITI: {trials_df['inter_trial_interval'].mean():.4f} seconds")
print(f"Median ITI: {trials_df['inter_trial_interval'].median():.4f} seconds")
print(f"Min ITI: {trials_df['inter_trial_interval'].min():.4f} seconds")
print(f"Max ITI: {trials_df['inter_trial_interval'].max():.4f} seconds")

# Plot inter-trial intervals
plt.figure(figsize=(14, 6))
plt.plot(trials_df.index[1:], trials_df['inter_trial_interval'][1:], 'o-', alpha=0.6)
plt.xlabel('Trial Number')
plt.ylabel('Inter-Trial Interval (seconds)')
plt.title('Inter-Trial Intervals')
plt.grid(True, alpha=0.3)
plt.savefig('explore/images/inter_trial_intervals.png')

# Analyze neural activity around trials for a few example units
print("\nAnalyzing neural activity around trials...")

# Get unit information
units = nwb.units
print(f"Number of units: {len(units.id)}")

# Select a few units with different firing rates
unit_ids = units.id[:]
unit_spike_counts = []

for i, unit_id in enumerate(unit_ids):
    spike_times = units.get_unit_spike_times(i)  # Use index i instead of unit_id
    unit_spike_counts.append(len(spike_times))

# Find high, medium, and low firing units
sorted_indices = np.argsort(unit_spike_counts)
high_firing_idx = sorted_indices[-1]
medium_firing_idx = sorted_indices[len(sorted_indices)//2]
low_firing_idx = sorted_indices[0]

selected_units = [high_firing_idx, medium_firing_idx, low_firing_idx]
print(f"Selected units for analysis (high, medium, low firing): {[unit_ids[idx] for idx in selected_units]}")

# Analyze spike times relative to trial starts for selected units
# We'll create peri-event time histograms (PETHs) around trial starts

def create_peth(spike_times, event_times, window=(-1, 3), bin_size=0.05):
    """Create a peri-event time histogram for spikes around events"""
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    counts = np.zeros((len(event_times), len(bins) - 1))
    
    for i, event in enumerate(event_times):
        # Align spikes to this event
        aligned_spikes = spike_times - event
        
        # Count spikes in each bin
        hist, _ = np.histogram(aligned_spikes, bins=bins)
        counts[i] = hist
    
    # Average across events
    average_counts = np.mean(counts, axis=0)
    bin_centers = bins[:-1] + bin_size/2
    
    return bin_centers, average_counts

# Get trial start times
trial_starts = trials_df['start_time'].values[:50]  # Just use the first 50 trials for faster computation
print(f"Using {len(trial_starts)} trials for PETH analysis")

# Plot PETHs for selected units
plt.figure(figsize=(14, 10))

for i, unit_idx in enumerate(selected_units):
    unit_id = unit_ids[unit_idx]
    spike_times = units.get_unit_spike_times(unit_idx)
    
    # Calculate PETH
    bin_centers, average_counts = create_peth(spike_times, trial_starts)
    
    # Convert to firing rate (spikes per second)
    bin_size = bin_centers[1] - bin_centers[0]
    firing_rate = average_counts / bin_size
    
    # Plot
    plt.subplot(3, 1, i+1)
    plt.bar(bin_centers, firing_rate, width=bin_size, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
    plt.xlabel('Time Relative to Trial Start (seconds)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_id} - Peri-Event Time Histogram')
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig('explore/images/peths_trial_start.png')

# Let's also look at trial-by-trial activity (spike rastergrams)
# for the high-firing unit around trial starts
plt.figure(figsize=(14, 8))
high_firing_unit_idx = selected_units[0]
high_firing_unit_id = unit_ids[high_firing_unit_idx]
spike_times = units.get_unit_spike_times(high_firing_unit_idx)

# Let's just look at the first 20 trials for clarity
num_trials_to_plot = 20
window = (-1, 3)  # 1 second before to 3 seconds after trial start

for i in range(num_trials_to_plot):
    trial_start = trials_df.iloc[i]['start_time']
    
    # Find spikes within the window around this trial start
    mask = (spike_times >= trial_start + window[0]) & (spike_times <= trial_start + window[1])
    aligned_spikes = spike_times[mask] - trial_start
    
    # Plot spikes for this trial
    plt.plot(aligned_spikes, np.ones_like(aligned_spikes) * i, '|', markersize=5, color='blue')

plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
plt.xlabel('Time Relative to Trial Start (seconds)')
plt.ylabel('Trial Number')
plt.title(f'Unit {high_firing_unit_id} - Spike Raster Around Trial Starts')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('explore/images/raster_trial_aligned.png')

# Finally, let's compare overall neural activity (spike counts per 1-second bin)
# with trial timing to look for any relationships
print("\nAnalyzing overall relationship between neural activity and trials...")

# Use the high firing unit for this analysis
unit_idx = high_firing_unit_idx
unit_id = unit_ids[unit_idx]
spike_times = units.get_unit_spike_times(unit_idx)

# Create bins for counting spikes
recording_end = max(trials_df['stop_time'].max(), spike_times.max())
bin_size = 1.0  # 1-second bins
bins = np.arange(0, recording_end + bin_size, bin_size)
spike_counts, _ = np.histogram(spike_times, bins=bins)
bin_centers = bins[:-1] + bin_size/2

plt.figure(figsize=(14, 10))

# Plot spike counts over time
plt.subplot(2, 1, 1)
plt.plot(bin_centers, spike_counts, alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Spike Count (per second)')
plt.title(f'Unit {unit_id} - Spike Counts Over Time')
plt.grid(True, alpha=0.3)

# On the same time axis, plot trial start/stop times as vertical spans
plt.subplot(2, 1, 2)
for i in range(len(trials_df)):
    start = trials_df.iloc[i]['start_time']
    stop = trials_df.iloc[i]['stop_time']
    plt.axvspan(start, stop, alpha=0.2, color='gray')

plt.xlabel('Time (seconds)')
plt.ylabel('Trials')
plt.title('Trial Timing')
plt.grid(True, alpha=0.3)
plt.yticks([])

# Set the x-axis to be the same for both subplots
max_time = max(recording_end, trials_df['stop_time'].max()) + 10
plt.xlim(0, max_time)
plt.subplot(2, 1, 1)
plt.xlim(0, max_time)

plt.tight_layout()
plt.savefig('explore/images/neuron_activity_vs_trials.png')

# Clean up
io.close()
print("\nScript completed successfully.")