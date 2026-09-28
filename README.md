# Apex AI - Motorsport Analytics & Race Strategy Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![FastF1](https://img.shields.io/badge/Data-FastF1-red.svg)](https://github.com/theOehrly/Fast-F1)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Apex AI** is a machine learning-powered motorsport analytics platform engineered for Formula 1 enthusiasts, strategists, and sim-racers. By ingesting high-frequency telemetry from official F1 sessions and custom sim-racing logs, Apex AI models non-linear tire degradation curves using Random Forest regression, compares driver telemetry to optimal session benchmarks, and delivers real-time strategic pit stop insights.

---

## Key Features

- **ML-Powered Tire Degradation Modeling**:
  - Filters out Safety Car (SC), Virtual Safety Car (VSC), and out-laps.
  - Trains a `RandomForestRegressor` on stint lap times vs. tire life to capture non-linear thermal degradation.
  - Automatically calculates degradation rate ($\Delta$ seconds/lap) and suggests tactical decisions (pit window opening, undercut viability, or overcut tire preservation).

- **Interactive GPS Track Maps & Zone Overlays**:
  - Reconstructs 2D track layout from driver coordinates ($X, Y$).
  - Overlays specific high-stress racing events: braking zones and full-throttle acceleration points.
  - Visualizes driver racing line alongside the session's overall fastest lap.

- **High-Resolution Telemetry Traces**:
  - Synchronized distance-based telemetry plots: **Speed (km/h)**, **Throttle %**, **Brake application**, and **Gear shifts**.

- **Driver Profiles & Chassis Specs**:
  - Dynamic driver headshots, official team livery color branding, and powertrain specs (chassis model, engine manufacturer, hybrid energy split, minimum weight regulations).

- **Race Classification Leaderboards**:
  - Live session finishing orders, team classifications, driver status (Finished, DNF, +Laps), and championship points awarded.

- **Sim-Racing Telemetry Uploader**:
  - Upload custom CSV telemetry from sim-racing titles (*iRacing, Assetto Corsa, F1 23/24, rFactor 2*).
  - Automatically runs regression to predict stint performance and degradation trends.

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────┐
│                   Web Dashboard UI                     │
│         (HTML5, Tailwind CSS, Chart.js, Jinja2)        │
└───────────────────────────┬────────────────────────────┘
                            │ REST API (JSON)
┌───────────────────────────▼────────────────────────────┐
│                    Flask API Server                    │
│                        (app.py)                        │
└──────┬────────────────────┬────────────────────┬───────┘
       │                    │                    │
┌──────▼───────┐     ┌──────▼───────┐     ┌──────▼───────┐
│   FastF1     │     │ Scikit-Learn │     │  CSV Parser  │
│  Telemetry   │     │Random Forest │     │ (Sim-Racing) │
│ & Lap Cache  │     │ ML Regressor │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Getting Started

### Prerequisites

- **Python 3.9+** installed on your system.
- `pip` (Python package manager).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sathwik-e/apex-ai.git
   cd apex-ai
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

1. **Launch the Flask server**:
   ```bash
   python app.py
   ```

2. **Open the Dashboard**:
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

> **Note on Telemetry Caching**:
> The application uses FastF1's built-in local cache (`f1_cache/`). When querying a session for the first time, FastF1 downloads the session data from official timing servers. Subsequent queries for the same Grand Prix will load instantly from the local cache.

---

## API Endpoints

### 1. Telemetry & Strategy Analysis
- **URL**: `/api/analyze`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "year": 2024,
    "circuit": "Monza",
    "driver": "VER",
    "compound": "MEDIUM"
  }
  ```
- **Response**: JSON containing `ml_data` (actual vs predicted tire degradation), `insight` (strategic recommendation), `driver_profile`, `car_specs`, `track_map` coordinates, and `telemetry_trace`.

### 2. Session Leaderboard
- **URL**: `/api/leaderboard`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "year": 2024,
    "circuit": "Monza"
  }
  ```
- **Response**: Array of driver finishing positions, points, status, and team names.

### 3. Custom Sim-Racing CSV Analysis
- **URL**: `/api/upload_csv`
- **Method**: `POST`
- **Body**: `multipart/form-data` with `file: <telemetry.csv>`
- **Expected CSV Format**:
  ```csv
  Lap,Time
  1,1:24.350
  2,1:24.510
  3,1:24.780
  4,1:25.120
  ```

---

## Project Structure

```text
apex-ai/
├── app.py                     # Main Flask application & ML pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Excludes caches, venv, and large telemetry files
├── index.html                 # Standalone web UI preview
├── Apex AI Data Fetcher.py    # Standalone FastF1 data collection script
├── apex ai .py                # Strategy simulation helper
├── apex ai ML.py              # ML model experimentation script
├── static/
│   └── cars/                  # High-resolution F1 team car renders
│       ├── Ferrari.png
│       ├── Mercedes.png
│       ├── mclaren.png
│       └── redbull.png
└── templates/
    └── index.html             # Full-featured responsive strategy dashboard
```

---

## Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **Motorsport Data Engine**: FastF1, Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest Regressor)
- **Frontend**: HTML5, Tailwind CSS, Chart.js

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
