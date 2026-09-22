from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import fastf1
import pandas as pd
import numpy as np
import io
import traceback
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

# Mapping for engine/chassis specs
team_specs = {
    "Red Bull Racing": {"chassis": "RB", "engine": "Honda RBPT"},
    "Ferrari": {"chassis": "SF", "engine": "Ferrari"},
    "Mercedes": {"chassis": "W", "engine": "Mercedes"},
    "McLaren": {"chassis": "MCL", "engine": "Mercedes"},
    "Aston Martin": {"chassis": "AMR", "engine": "Mercedes"},
    "Alpine": {"chassis": "A", "engine": "Renault"},
    "Williams": {"chassis": "FW", "engine": "Mercedes"},
    "RB": {"chassis": "VCARB", "engine": "Honda RBPT"},
    "Kick Sauber": {"chassis": "C", "engine": "Ferrari"},
    "Haas F1 Team": {"chassis": "VF", "engine": "Ferrari"}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        year = int(data['year'])
        circuit = data['circuit']
        driver = data['driver'].upper()
        compound = data['compound'].upper()

        # 1. Load Session
        session = fastf1.get_session(year, circuit, 'R')
        session.load(telemetry=True, weather=False, messages=False)

        # 2. Get Driver Laps & Clean Data
        driver_laps = session.laps.pick_driver(driver)
        
        if driver_laps.empty:
            return jsonify({"error": f"No laps found for driver {driver}."}), 400

        quick_laps = driver_laps.pick_quicklaps()
        compound_laps = quick_laps[quick_laps['Compound'] == compound]
        
        # Remove SC/VSC laps
        valid_laps = compound_laps[compound_laps['TrackStatus'] == '1']
        valid_laps = valid_laps.dropna(subset=['TyreLife', 'Compound', 'LapTime'])

        if len(valid_laps) < 3:
            return jsonify({"error": f"Not enough clean laps found for {driver} on {compound} tires to run analysis. Try a different compound."}), 400

        # Convert LapTime to seconds
        valid_laps['LapTime_sec'] = valid_laps['LapTime'].dt.total_seconds()

        # 3. Machine Learning (Random Forest)
        X = valid_laps[['TyreLife']].values
        y = valid_laps['LapTime_sec'].values
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        X_pred = np.arange(valid_laps['TyreLife'].min(), valid_laps['TyreLife'].max() + 1).reshape(-1, 1)
        y_pred = model.predict(X_pred)

        avg_degradation = float((y_pred[-1] - y_pred[0]) / len(y_pred))

        # AI Insight Logic
        if avg_degradation > 0.15:
            insight = f"Critical wear detected (+{avg_degradation:.3f}s / Lap). Tires are falling off the cliff. Immediate pit stop recommended."
        elif avg_degradation > 0.06:
            insight = f"Moderate wear (+{avg_degradation:.3f}s / Lap). Standard pit window active. The undercut is highly viable if stuck in dirty air."
        else:
            insight = f"Excellent tire preservation (+{avg_degradation:.3f}s / Lap). Driver is extending tire life beautifully. I suggest overcutting rivals to gain track position."

        # 4. Telemetry Track Map (Driver vs Optimal)
        driver_fastest = driver_laps.pick_fastest()
        session_fastest = session.laps.pick_fastest()
        
        # Safely parse the fastest lap time to standard format (e.g. 1:25.123)
        try:
            lap_time_obj = driver_fastest['LapTime']
            if pd.isnull(lap_time_obj):
                fastest_lap_str = "N/A"
            else:
                minutes = int(lap_time_obj.total_seconds() // 60)
                seconds = lap_time_obj.total_seconds() % 60
                fastest_lap_str = f"{minutes}:{seconds:06.3f}"
        except:
            fastest_lap_str = "N/A"
        
        driver_tel = driver_fastest.get_telemetry()
        session_tel = session_fastest.get_telemetry()

        # Extract braking and full throttle points
        braking_points = driver_tel[driver_tel['Brake'] == True]
        throttle_points = driver_tel[driver_tel['Throttle'] >= 99]

        # 5. Get Driver Info & Car Specs
        driver_info = session.get_driver(driver)
        team_name = driver_info['TeamName']
        color = f"#{driver_info['TeamColor']}"
        
        # Dynamically fetch the official driver headshot
        headshot_url = driver_info.get('HeadshotUrl', '')
        if pd.isna(headshot_url) or not headshot_url:
            headshot_url = "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/2col/image.png"
        
        specs = team_specs.get(team_name, {"chassis": "Unknown", "engine": "Unknown"})
        chassis = f"{specs['chassis']}-{str(year)[-2:]}"
        engine = specs['engine']

        car_details = {
            "chassis": chassis,
            "engine": engine,
            "hp": "~1,050+ HP (Combined)",
            "weight": "798 kg (Minimum)",
            "torque": "~700+ Nm",
            "split": "80% ICE / 20% Electric",
            "battery": "4 MJ / Lap (Regulated)"
        }

        # Structure response
        response = {
            "ml_data": {
                "actual_x": X.flatten().tolist(),
                "actual_y": y.tolist(),
                "pred_x": X_pred.flatten().tolist(),
                "pred_y": y_pred.tolist(),
                "avg_deg": avg_degradation,
                "fastest_lap": fastest_lap_str
            },
            "insight": insight,
            "driver_profile": {
                "team": team_name,
                "color": color,
                "headshot_url": headshot_url
            },
            "car_specs": car_details,
            "track_map": {
                "driver_x": driver_tel['X'].tolist(),
                "driver_y": driver_tel['Y'].tolist(),
                "optimal_x": session_tel['X'].tolist(),
                "optimal_y": session_tel['Y'].tolist(),
                "brake_x": braking_points['X'].tolist(),
                "brake_y": braking_points['Y'].tolist(),
                "throttle_x": throttle_points['X'].tolist(),
                "throttle_y": throttle_points['Y'].tolist(),
                "start_x": [driver_tel['X'].iloc[0]] if not driver_tel.empty else [0],
                "start_y": [driver_tel['Y'].iloc[0]] if not driver_tel.empty else [0]
            },
            "telemetry_trace": {
                "distance": driver_tel['Distance'].tolist(),
                "speed": driver_tel['Speed'].tolist(),
                "throttle": driver_tel['Throttle'].tolist(),
                "brake": [100 if b else 0 for b in driver_tel['Brake'].tolist()],
                "gear": driver_tel['nGear'].tolist()
            }
        }
        return jsonify(response)

    except Exception as e:
        print("\n--- ERROR CAUGHT IN BACKEND ---")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/leaderboard', methods=['POST'])
def leaderboard():
    data = request.json
    try:
        session = fastf1.get_session(int(data['year']), data['circuit'], 'R')
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results
        
        leaderboard_data = []
        for index, row in results.iterrows():
            leaderboard_data.append({
                "position": row['Position'],
                "driver": row['Abbreviation'],
                "team": row['TeamName'],
                "status": row['Status'],
                "points": row['Points']
            })
        return jsonify(leaderboard_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        
        lap_col = next((col for col in df.columns if 'lap' in col.lower()), None)
        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
        
        if not lap_col or not time_col:
            return jsonify({"error": "CSV must contain a 'Lap' and 'Time' column."}), 400
            
        if df[time_col].dtype == object:
            df['LapTime_sec'] = df[time_col].apply(lambda x: float(str(x).split(':')[-1]) if ':' in str(x) else float(x))
        else:
            df['LapTime_sec'] = df[time_col]
            
        df = df.dropna(subset=[lap_col, 'LapTime_sec'])
            
        X = df[[lap_col]].values
        y = df['LapTime_sec'].values
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        X_pred = np.arange(df[lap_col].min(), df[lap_col].max() + 1).reshape(-1, 1)
        y_pred = model.predict(X_pred)
        avg_degradation = float((y_pred[-1] - y_pred[0]) / len(y_pred))
        
        return jsonify({
            "actual_x": X.flatten().tolist(),
            "actual_y": y.tolist(),
            "pred_x": X_pred.flatten().tolist(),
            "pred_y": y_pred.tolist(),
            "avg_deg": avg_degradation
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)