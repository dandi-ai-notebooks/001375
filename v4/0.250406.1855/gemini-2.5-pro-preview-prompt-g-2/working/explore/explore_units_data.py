# explore/explore_units_data.py
# This script loads the NWB file, explores the units data (spike times),
# and creates a raster plot for a few units.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting script: explore_units_data.py")

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
try:
    print(f"Attempting to load NWB file from: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    if nwb.units is not None:
        units_table = nwb.units
        print("Units table found.")

        try:
            units_df = units_table.to_dataframe()
            print(f"Units DataFrame shape: {units_df.shape}")
            print("First 5 units (info):")
            print(units_df.head())

            num_units_to_plot = min(5, len(units_df))
            if num_units_to_plot > 0:
                selected_unit_ids = units_df.index[:num_units_to_plot]
                print(f"\nSelected unit IDs for plotting: {selected_unit_ids.tolist()}")

                spike_times_list = []
                for unit_id in selected_unit_ids:
                    # In NWB, spike_times for a unit_id is accessed via the table directly
                    # The .to_dataframe() method already puts spike_times into series of arrays
                    spikes = units_df.loc[unit_id, 'spike_times']
                    spike_times_list.append(spikes)
                
                sns.set_theme()
                plt.figure(figsize=(15, 8))
                colors = plt.cm.get_cmap('viridis', num_units_to_plot)
                
                # Using eventplot for raster plot
                plt.eventplot(spike_times_list, linelengths=0.75, colors=[colors(i) for i in range(num_units_to_plot)])

                plt.yticks(np.arange(num_units_to_plot), [f"Unit {uid}" for uid in selected_unit_ids])
                plt.xlabel("Time (s)")
                plt.ylabel("Unit ID")
                plt.title(f"Spike Raster Plot (First {num_units_to_plot} Units)")
                
                # Determine x-axis limits. If trials exist, use the end of the first few trials.
                # Otherwise, use a fixed limit like 60s or max spike time if shorter.
                max_time_limit = 60.0 # seconds
                if nwb.trials is not None and len(nwb.trials.stop_time) > 0:
                    # Consider up to the end of the 5th trial or max_time_limit, whichever is smaller
                    trial_end_time_limit = nwb.trials.stop_time[min(4, len(nwb.trials.stop_time)-1)]
                    current_xlim_upper = min(max_time_limit, trial_end_time_limit)
                else:
                    # If no trials, find max spike time among plotted units, cap at max_time_limit
                    all_plotted_spikes = np.concatenate(spike_times_list) if len(spike_times_list) > 0 else np.array([0])
                    max_spike_t = np.max(all_plotted_spikes) if len(all_plotted_spikes) > 0 else max_time_limit
                    current_xlim_upper = min(max_time_limit, max_spike_t if max_spike_t > 0 else max_time_limit)

                plt.xlim(0, current_xlim_upper)
                plt.grid(True, axis='x', linestyle=':', alpha=0.7)
                
                plot_filename = "explore/spike_raster_plot.png"
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
                plt.close()

            else:
                print("No units available to plot.")

        except Exception as e_df:
            print(f"Error processing units data or plotting: {e_df}")
            import traceback
            traceback.print_exc()
            
    else:
        print("No 'units' table found in NWB file.")

    print("Closing NWB file resources...")
    io.close()

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Script finished.")