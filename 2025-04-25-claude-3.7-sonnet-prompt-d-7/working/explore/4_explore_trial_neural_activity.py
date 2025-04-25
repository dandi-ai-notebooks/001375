# Script to explore neural activity in relation to trials/laps

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial information
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']

# Select a subset of units to analyze
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")
# Choose 5 units to analyze
unit_indices = np.linspace(0, len(units_df)-1, 5, dtype=int)
print(f"Analyzing units at indices: {unit_indices}")

# Create a plot for population activity during trials
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# 1. Plot trial durations across experiment timeline
trial_starts = trials_df['start_time']
trial_durations = trials_df['duration']
axes[0].scatter(trial_starts/60, trial_durations, alpha=0.7)
axes[0].set_xlabel('Time (minutes)')
axes[0].set_ylabel('Trial Duration (s)')
axes[0].set_title('Trial Durations Over Time')
axes[0].grid(True)

# 2. Average firing rate within each trial for selected units
trial_firing_rates = np.zeros((len(trials_df), len(unit_indices)))

for i, unit_idx in enumerate(unit_indices):
    spike_times = nwb.units['spike_times'][unit_idx]
    for j, (start, end) in enumerate(zip(trials_df['start_time'], trials_df['stop_time'])):
        # Count spikes in this trial
        trial_spikes = spike_times[(spike_times >= start) & (spike_times <= end)]
        trial_duration = end - start
        # Calculate firing rate (Hz)
        firing_rate = len(trial_spikes) / trial_duration if trial_duration > 0 else 0
        trial_firing_rates[j, i] = firing_rate

# Plot average firing rate within trials across experiment
for i, unit_idx in enumerate(unit_indices):
    axes[1].plot(trials_df['start_time']/60, trial_firing_rates[:, i], 
                 label=f'Unit {unit_idx}', alpha=0.7)
axes[1].set_xlabel('Time (minutes)')
axes[1].set_ylabel('Firing Rate (Hz)')
axes[1].set_title('Average Firing Rate Within Each Trial')
axes[1].legend()
axes[1].grid(True)

# 3. Create a peri-event time histogram (PETH) around trial start
# Select 3 units for clarity
peth_unit_indices = unit_indices[:3]
bin_size = 0.5  # 500 ms
window = (-2, 10)  # 2 seconds before to 10 seconds after trial start
bins = np.arange(window[0], window[1] + bin_size, bin_size)
bin_centers = (bins[:-1] + bins[1:]) / 2

peth_matrix = np.zeros((len(peth_unit_indices), len(bin_centers)))

for i, unit_idx in enumerate(peth_unit_indices):
    spike_times = nwb.units['spike_times'][unit_idx]
    all_aligned_spikes = []
    
    # Only use first 100 trials for faster processing
    for j, start in enumerate(trials_df['start_time'][:100]):
        # Align spikes to trial start
        aligned_spikes = spike_times - start
        # Select spikes within window
        window_spikes = aligned_spikes[(aligned_spikes >= window[0]) & (aligned_spikes < window[1])]
        all_aligned_spikes.extend(window_spikes)
    
    # Count spikes in bins
    counts, _ = np.histogram(all_aligned_spikes, bins=bins)
    # Convert to firing rate
    peth_matrix[i, :] = counts / (bin_size * min(100, len(trials_df)))

# Plot PETH
for i, unit_idx in enumerate(peth_unit_indices):
    axes[2].plot(bin_centers, peth_matrix[i, :], label=f'Unit {unit_idx}')
axes[2].axvline(x=0, color='r', linestyle='--', label='Trial Start')
axes[2].set_xlabel('Time from Trial Start (s)')
axes[2].set_ylabel('Firing Rate (Hz)')
axes[2].set_title('Peri-Event Time Histogram (PETH) Around Trial Start')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('explore/trial_neural_activity.png')

# Now examine the relationship between trial duration and neural activity
plt.figure(figsize=(12, 8))

# Calculate average firing rate for each unit in each trial
# Use up to 200 trials to avoid overcrowding the plot
n_trials = min(200, len(trials_df))
firing_duration_matrix = np.zeros((n_trials, len(unit_indices)))

for i, unit_idx in enumerate(unit_indices):
    spike_times = nwb.units['spike_times'][unit_idx]
    for j, (start, end) in enumerate(zip(trials_df['start_time'][:n_trials], trials_df['stop_time'][:n_trials])):
        # Count spikes in this trial
        trial_spikes = spike_times[(spike_times >= start) & (spike_times <= end)]
        trial_duration = end - start
        # Calculate firing rate (Hz)
        firing_rate = len(trial_spikes) / trial_duration if trial_duration > 0 else 0
        firing_duration_matrix[j, i] = firing_rate

# Plot the relationship between trial duration and firing rate for each unit
# Create a grid of scatter plots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, unit_idx in enumerate(unit_indices):
    if i < len(axs):
        axs[i].scatter(trials_df['duration'][:n_trials], firing_duration_matrix[:, i], alpha=0.6)
        axs[i].set_xlabel('Trial Duration (s)')
        axs[i].set_ylabel('Firing Rate (Hz)')
        axs[i].set_title(f'Unit {unit_idx}')
        axs[i].grid(True)

# Remove any unused subplots
for i in range(len(unit_indices), len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.savefig('explore/trial_duration_vs_firing.png')

# Close the file
io.close()
h5_file.close()
remote_file.close()