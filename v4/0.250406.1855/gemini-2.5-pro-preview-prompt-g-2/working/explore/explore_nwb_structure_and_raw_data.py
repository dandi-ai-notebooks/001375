# explore/explore_nwb_structure_and_raw_data.py
# This script loads the NWB file, prints some basic info,
# and plots a small segment of raw ephys data.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting script: explore_nwb_structure_and_raw_data.py")

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
try:
    print(f"Attempting to load NWB file from: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r') # Ensure read-only mode
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read-only mode for io
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Print basic information
    print(f"Session description: {nwb.session_description}")
    print(f"Identifier: {nwb.identifier}")
    print(f"Session start time: {nwb.session_start_time}")

    # Print acquisition keys
    print(f"Acquisition keys: {list(nwb.acquisition.keys())}")

    if "time_series" in nwb.acquisition:
        ts = nwb.acquisition["time_series"]
        print(f"Time series data shape: {ts.data.shape}")
        print(f"Time series rate: {ts.rate} Hz")

        # Plot a small segment of raw data
        if ts.data.shape[0] > 0 and ts.data.shape[1] > 0:
            duration_to_plot_sec = 1.0  # seconds
            channel_to_plot_idx = 0   # first channel

            num_samples_to_plot = int(ts.rate * duration_to_plot_sec)

            if ts.data.shape[0] >= num_samples_to_plot:
                print(f"Plotting {duration_to_plot_sec}s of data from channel {channel_to_plot_idx}")
                raw_data_segment = ts.data[0:num_samples_to_plot, channel_to_plot_idx]

                # Create time vector
                time_vector = np.linspace(0, duration_to_plot_sec, num_samples_to_plot, endpoint=False)

                # Plotting
                sns.set_theme() # Apply seaborn styling
                plt.figure(figsize=(12, 6))
                plt.plot(time_vector, raw_data_segment)
                plt.title(f"Raw Ephys Data - Channel {channel_to_plot_idx} (First {duration_to_plot_sec}s)")
                plt.xlabel("Time (s)")
                plt.ylabel(f"Amplitude ({ts.unit})")
                plt.grid(True)
                
                plot_filename = "explore/raw_ephys_trace.png"
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
                plt.close() # Close the figure to free memory
            else:
                print(f"Not enough samples to plot {duration_to_plot_sec}s of data. Available samples: {ts.data.shape[0]}")
        else:
            print("Time series data is empty or has no channels.")
    else:
        print("No 'time_series' found in nwb.acquisition.")

    print("Closing NWB file resources...")
    io.close() # closes h5_file as well if it was opened by NWBHDF5IO
    # remote_file.close() # remfile does not have an explicit close for read-only

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Script finished.")