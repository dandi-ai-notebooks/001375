# explore/explore_trials_data.py
# This script loads the NWB file, explores the trials data,
# and plots a histogram of trial durations.

import pynwb
import h5py
import remfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting script: explore_trials_data.py")

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
try:
    print(f"Attempting to load NWB file from: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    if "trials" in nwb.intervals:
        trials_table = nwb.intervals["trials"]
        print("Trials table found.")

        # Convert to DataFrame
        # Note: The nwb_file_info output suggests trials.to_dataframe() is available.
        # Need to be careful as direct access to start_time and stop_time as VectorData is also shown.
        # We'll try to_dataframe() first.
        try:
            trials_df = trials_table.to_dataframe()
            print("Trials data converted to DataFrame successfully.")
            print("First 5 trials:")
            print(trials_df.head())

            if 'start_time' in trials_df.columns and 'stop_time' in trials_df.columns:
                # Calculate trial durations
                trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
                print("\nTrial durations calculated. First 5 durations:")
                print(trials_df['duration'].head())

                # Plot histogram of trial durations
                sns.set_theme()
                plt.figure(figsize=(10, 6))
                sns.histplot(trials_df['duration'], kde=False, bins=30)
                plt.title("Distribution of Trial Durations")
                plt.xlabel("Duration (s)")
                plt.ylabel("Number of Trials")
                plt.grid(True)
                
                plot_filename = "explore/trial_durations_hist.png"
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
                plt.close()
            else:
                print("Could not find 'start_time' or 'stop_time' columns in the trials DataFrame.")
                print(f"Available columns: {trials_df.columns}")

        except Exception as e_df:
            print(f"Error converting trials to DataFrame or plotting: {e_df}")
            # Fallback or further investigation might be needed if to_dataframe() fails.
            # For now, just print the error.
            # print(f"Trials table columns: {trials_table.colnames}")
            # start_times = trials_table['start_time'][:]
            # stop_times = trials_table['stop_time'][:]
            # print(f"Number of start times: {len(start_times)}, Number of stop times: {len(stop_times)}")


    else:
        print("No 'trials' interval table found in nwb.intervals.")

    print("Closing NWB file resources...")
    io.close()

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Script finished.")