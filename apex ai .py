import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# FastF1 downloads a lot of data. We enable caching so it saves the data locally
# and doesn't take 5 minutes to download every time you run the script.
# Make sure to create a folder named 'f1_cache' in the same directory as this script.
fastf1.Cache.enable_cache('f1_cache')

def get_fastest_lap_telemetry(year, track, session_type):
    """
    Fetches the fastest lap telemetry for a given race.
    """
    print(f"Loading data for {year} {track} {session_type}...")
    
    # This connects to the F1 API and downloads the timing data
    session = fastf1.get_session(year, track, session_type)
    session.load()
    
    # Get the single fastest lap of the entire session
    fastest_lap = session.laps.pick_fastest()
    driver = fastest_lap['Driver']
    lap_time = fastest_lap['LapTime']
    
    print(f"\nFastest Lap by: {driver}")
    print(f"Lap Time: {lap_time}")
    
    # This gives us a Pandas DataFrame with data recorded multiple times per second
    telemetry = fastest_lap.get_telemetry()
    
    print("\n--- Telemetry Sample (First 5 data points) ---")
    # We only print the columns we care about for ML later
    print(telemetry[['Time', 'Speed', 'nGear', 'Throttle', 'Brake']].head())
    
    return telemetry, driver

def plot_telemetry(telemetry, driver):
    """
    Visualizes the speed and throttle data for the lap.
    This is perfect for the "Exploratory Data Analysis" section of your case study!
    """
    print("\nGenerating telemetry graph...")
    
    # Create a plot with 2 subplots (Speed on top, Throttle on bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Plot Speed
    ax1.plot(telemetry['Time'].dt.total_seconds(), telemetry['Speed'], color='cyan', label=f'{driver} Speed')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title(f'{driver} - Fastest Lap Telemetry')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Throttle
    ax2.plot(telemetry['Time'].dt.total_seconds(), telemetry['Throttle'], color='lime', label=f'{driver} Throttle')
    ax2.set_ylabel('Throttle %')
    ax2.set_xlabel('Time (Seconds)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Show the graph
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Let's look at the 2023 Monza (Italy) Grand Prix race session
    # You can change these variables to look at different races!
    try:
        data, driver_name = get_fastest_lap_telemetry(2023, 'Monza', 'R')
        plot_telemetry(data, driver_name)
        print("\nSuccess! Apex AI has successfully extracted and graphed F1 telemetry.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Did you remember to create the 'f1_cache' folder?")