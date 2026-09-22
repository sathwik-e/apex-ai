import fastf1
import pandas as pd
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('f1_cache') #using cache to quickly retrieve data

def get_fastest_lap_telemetry(year, track, session_type):
    #fetches data
    print(f"Loading data for {year} {track} {session_type}...")
    #this connects to the api and fetches data
    session = fastf1.get_session(year, track, session_type)
    session.load()

    #need to get the fastest lap of the entire session
    fastest_lap = session.laps.pick_fastest()
    driver = fastest_lap['Driver']
    lap_time = fastest_lap['LapTime']

    print(f"\nFastest Lap by: {driver}")
    print(f"Lap Time: {lap_time}")

    #this gives us the telemetry for the fastest lap
    telemetry = fastest_lap.get_telemetry()

    print("\n---- Telemetry Sample (First 5 data points) ----")
    #we only print the columns we need for ML 
    print(telemetry[['Time', 'Speed', 'nGear', 'Throttle', 'Brake']].head())

    return telemetry, driver

def plot_telemetry(telemetry, driver):
    #visualizes the telemetry data
    print("\nGenerating telemetry graph...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    #plot speed
    ax1.plot(telemetry['Time'].dt.total_seconds(), telemetry['Speed'], color='cyan', label=f'{driver} Speed')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title(f'{driver} - Fastest Lap Telemetry')
    ax1.legend()
    ax1.grid(True, linestyle='--',alpha=0.7)

    #plot throttle
    ax2.plot(telemetry['Time'].dt.total_seconds(), telemetry['Throttle'], color='green', label=f'{driver} Throttle')
    ax2.set_ylabel('Throttle %')
    ax2.set_xlabel('Time (Seconds)')
    ax2.legend()
    ax2.grid(True, linestyle='--',alpha=0.7)

    #to show the graph
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        data, driver_name = get_fastest_lap_telemetry(2023, 'Monza', 'R')
        plot_telemetry(data, driver_name)
        print("\nSuccessfully fetched and graphed telemetry data.")
    except Exception as e:
        print(f"An error occurred: {e}")