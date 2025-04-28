# Explore trial durations from the trials TimeIntervals table
# Goal: Load trial start and stop times, calculate durations,
# and plot a histogram of these durations.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Ensure the output directory exists
os.makedirs('explore', exist_ok=True)

print("Loading NWB file...")
# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Explicitly read-only
nwb = io.read()
print("NWB file loaded.")

# Access the trials table
if 'trials' in nwb.intervals:
    trials_table = nwb.intervals['trials']
    print("Trials table found.")

    # Convert to DataFrame
    trials_df = trials_table.to_dataframe()
    print(f"Converted trials table to DataFrame with {len(trials_df)} trials.")
    print("Trials DataFrame head:")
    print(trials_df.head())

    # Calculate trial durations
    if 'start_time' in trials_df.columns and 'stop_time' in trials_df.columns:
        trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
        print("Calculated trial durations.")

        # Plot histogram of durations
        print("Generating histogram of trial durations...")
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(trials_df['duration'], ax=ax, kde=False) # Plot histogram using seaborn

        ax.set_xlabel('Trial Duration (s)')
        ax.set_ylabel('Number of Trials')
        ax.set_title('Distribution of Trial Durations')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        output_path = 'explore/trial_durations_hist.png'
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        print("Could not find 'start_time' or 'stop_time' columns in trials DataFrame.")

else:
    print("No 'trials' interval table found in the NWB file.")

# Close resources
io.close()
# remote_file.close()

print("Script finished.")