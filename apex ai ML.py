import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#enabling cache just like the fetcher
fastf1.Cache.enable_cache('f1_cache')

def train_tire_degradation_model(year, track, session_type):
    #fetches data
    print(f"Loading data for {year} {track} {session_type}...")
    session = fastf1.get_session(year, track, session_type)
    session.load()

    #get all laps
    laps = session.laps
    #fastf1 has a function to remove abnormally slow laps
    quick_laps = laps.pick_quicklaps().copy()
    #drop rows where we're missing imp stuff where tire data or LapTimes
    quick_laps = quick_laps.dropna(subset=['TyreLife', 'Compound', 'LapTime' ])
    # Machine learning models only understand numbers, not time objects
    #Convert LapTime into total seconds
    quick_laps['LapTime_sec'] = quick_laps['LapTime'].dt.total_seconds()

    #ML models dont understand words like soft we need dto encode them to numbers
    #SOFT = 0, MEDIUM = 1, HARD = 2
    compound_mapping = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    quick_laps['Compound_Num'] = quick_laps['Compound'].map(compound_mapping)
    quick_laps = quick_laps.dropna(subset=['Compound_Num'])

    print("Training the Random Forest model...")
    #features (X) are what we use to predict the target (y)
    X = quick_laps[['TyreLife', 'Compound_Num']]
    y = quick_laps['LapTime_sec']

    #Initialize the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Model training complete.")

    return model, quick_laps

def plot_predictions(model, clean_laps, driver):
    """ Graphs the actual lap times of a specific driver against what the AI *predicted* 
    they would run based on their tires"""

    print(f"\nGenerating prediction graph for {driver}...")

    # Get the laps for our specific driver
    driver_laps = clean_laps.pick_driver(driver)

    #Use the AI model to predict what lap time they should have Gotten
    driver_X = driver_laps[['TyreLife', 'Compound_Num']]
    predicted_LapTimes = model.predict(driver_X)

    #Plotting
    plt.figure(figsize=(10, 6))

    #1 Plot actual lap times as scatter dots
    plt.scatter(driver_laps['LapNumber'], driver_laps['LapTime_sec'], color='cyan', label=f'{driver} Actual Laps', alpha=0.7)

    #2 Plot the predicted treadline as a line graph
    plt.plot(driver_laps['LapNumber'], predicted_LapTimes, color='red', linewidth=2, label=f'{driver} Predicted Laps')

    plt.title(f'Apex AI Tire Degradation Prediction - {driver}')
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (Seconds)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #1 train the model on the entire monza grid
    ai_model, processed_data = train_tire_degradation_model(2023, 'Monza', 'R')
    #2 plot predictions for MAX VERSTAPPEN (VER)
    plot_predictions(ai_model, processed_data, 'VER')

    print("\nSuccess! Apex AI is now predicting")
