# This script explores the neural activity in the dataset
# focusing on spike times and unit activity patterns

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print("Loading NWB file from:", url)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get basic information about units
print(f"Number of units: {len(nwb.units.id)}")
print(f"Unit columns: {nwb.units.colnames}")

# Count the number of spikes for each unit
spike_counts = []
unit_ids = []

for i, unit_id in enumerate(nwb.units.id):
    spike_times = nwb.units['spike_times'][i]
    unit_ids.append(unit_id)
    spike_counts.append(len(spike_times))

# Create a dataframe with unit info
unit_df = pd.DataFrame({
    'unit_id': unit_ids,
    'spike_count': spike_counts
})

# Print summary statistics
print("\nSpike count statistics:")
print(f"Mean: {unit_df['spike_count'].mean():.1f}")
print(f"Median: {unit_df['spike_count'].median():.1f}")
print(f"Min: {unit_df['spike_count'].min()}")
print(f"Max: {unit_df['spike_count'].max()}")

# Create histogram of spike counts
plt.figure(figsize=(10, 6))
plt.hist(unit_df['spike_count'], bins=20, alpha=0.7, color='purple')
plt.xlabel('Number of Spikes')
plt.ylabel('Count (Units)')
plt.title('Distribution of Spike Counts Across Units')
plt.grid(True, alpha=0.3)
plt.savefig('explore/spike_count_distribution.png', dpi=300)
print("Saved spike count distribution to explore/spike_count_distribution.png")

# Explore firing rates over time for a few example units
plt.figure(figsize=(12, 8))

# Select top 5 most active units
top_units = unit_df.sort_values('spike_count', ascending=False).head(5)
print("\nTop 5 most active units:")
print(top_units)

# Time window (seconds)
bin_size = 10  # seconds per bin
max_time = 5000  # max time to consider in seconds

# Create a simple raster plot for a single unit
unit_index = top_units.index[0]  # Get the index of the most active unit
spike_times = nwb.units['spike_times'][unit_index]

plt.figure(figsize=(12, 4))
plt.eventplot([spike_times[:1000]], lineoffsets=[0], linelengths=[0.5], 
              colors=['black'])
plt.xlabel('Time (s)')
plt.title(f'Spike Raster for Unit {top_units.iloc[0]["unit_id"]} (first 1000 spikes)')
plt.yticks([])
plt.grid(True, alpha=0.3)
plt.xlim(0, min(max(spike_times[:1000]) + 10, 600))
plt.savefig('explore/spike_raster.png', dpi=300)
print("Saved spike raster plot to explore/spike_raster.png")

# Create firing rate plots for top units
plt.figure(figsize=(12, 10))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for i, (idx, unit) in enumerate(top_units.iterrows()):
    spike_times = nwb.units['spike_times'][idx]
    
    # Create histogram of spike times
    bins = np.arange(0, max_time, bin_size)
    counts, _ = np.histogram(spike_times, bins=bins)
    firing_rates = counts / bin_size  # Convert to spikes per second
    
    plt.subplot(len(top_units), 1, i+1)
    plt.plot(bins[:-1], firing_rates, color=colors[i], linewidth=1.5)
    plt.title(f'Unit {unit["unit_id"]} (Total spikes: {unit["spike_count"]})')
    plt.ylabel('Firing Rate (Hz)')
    
    if i == len(top_units) - 1:
        plt.xlabel('Time (s)')
    
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('explore/firing_rates.png', dpi=300)
print("Saved firing rate plots to explore/firing_rates.png")

# Create a plot to show relationship between spikes and trials
# for the most active unit
most_active_unit_idx = top_units.index[0]
most_active_unit_spikes = nwb.units['spike_times'][most_active_unit_idx]

# Get trial data
trials_df = nwb.trials.to_dataframe()

# Create a plot showing spike density in relation to trials
plt.figure(figsize=(12, 6))

# Plot trial periods as shaded regions
for _, trial in trials_df.head(50).iterrows():  # Plot first 50 trials for clarity
    plt.axvspan(trial['start_time'], trial['stop_time'], alpha=0.2, color='lightgray')

# Plot spike times
plt.scatter(most_active_unit_spikes[:5000], 
           np.ones_like(most_active_unit_spikes[:5000]), 
           s=1, color='black', alpha=0.5)

plt.xlabel('Time (s)')
plt.ylabel('Spike Presence')
plt.title(f'Spikes vs Trial Periods (Unit {top_units.iloc[0]["unit_id"]}, first 50 trials)')
plt.grid(True, alpha=0.3)
plt.yticks([])

# Limit to first 600 seconds for visibility
plt.xlim(0, 600)
plt.savefig('explore/spikes_vs_trials.png', dpi=300)
print("Saved spikes vs trials plot to explore/spikes_vs_trials.png")