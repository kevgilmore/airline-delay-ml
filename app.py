"""Flask app for flight delay predictions."""

import hashlib
from datetime import date, timedelta

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

app = Flask(__name__)

# Load model and metadata once at startup
_model = joblib.load("models/model.joblib")
_meta = joblib.load("models/model_metadata.joblib")
COLUMNS = _meta["columns"]
CATEGORY_VALUES = _meta["category_values"]
NUMERIC_DEFAULTS = _meta["numeric_defaults"]
ROUTE_STATS = _meta["route_stats"]
ROUTE_AIRLINES = _meta["route_airlines"]
FLIGHTS = _meta["flights"]

# Derived lookups for cascading dropdowns
_route_airlines_js = {f"{k[0]}||{k[1]}": v for k, v in ROUTE_AIRLINES.items()}
_airport_countries = {}
for airport, country in ROUTE_AIRLINES:
    _airport_countries.setdefault(airport, []).append(country)
_airport_countries = {k: sorted(set(v)) for k, v in _airport_countries.items()}



def build_prediction_row(form_data):
    """Build a single-row DataFrame matching the trained model's column schema."""
    row = pd.DataFrame(0, index=[0], columns=COLUMNS)

    airport = form_data["reporting_airport"]
    country = form_data["origin_destination_country"]

    stats = ROUTE_STATS.get((airport, country), NUMERIC_DEFAULTS)
    for feat in NUMERIC_FEATURES:
        row[feat] = stats[feat]

    for feat in CATEGORICAL_FEATURES:
        val = form_data[feat]
        col_name = f"{feat}_{val}"
        if col_name in COLUMNS:
            row[col_name] = 1

    return row


@app.route("/")
def index():
    return render_template(
        "index.html",
        airports=CATEGORY_VALUES["reporting_airport"],
        airport_countries=_airport_countries,
        route_airlines=_route_airlines_js,
    )


@app.route("/api/flights")
def api_flights():
    """Return generated flight schedule for a route selection."""
    airport = request.args.get("airport", "")
    country = request.args.get("country", "")
    airline = request.args.get("airline", "")
    direction = request.args.get("direction", "D")

    key = f"{airport}||{country}||{airline}||{direction}"
    dests = FLIGHTS.get(key, [])

    today = date.today()
    schedule = []

    for dest_info in dests:
        dest = dest_info["destination"]
        monthly = dest_info["flights_per_month"]
        daily = max(1, round(monthly / 30))
        code = airline[:2].upper().replace(" ", "")

        for day_offset in range(7):
            flight_date = today + timedelta(days=day_offset)
            for i in range(daily):
                # Deterministic time and flight number from route seed
                seed = f"{airport}-{dest}-{airline}-{direction}-{i}"
                h = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
                hour = 6 + (h % 16)
                minute = (h >> 4) % 4 * 15
                num = 100 + (int(hashlib.md5(seed.encode()).hexdigest()[:6], 16) % 900)

                schedule.append(
                    {
                        "flight_number": f"{code}{num}",
                        "destination": dest,
                        "date": flight_date.isoformat(),
                        "time": f"{hour:02d}:{minute:02d}",
                        "avg_delay": dest_info["avg_delay"],
                    }
                )

    schedule.sort(key=lambda x: (x["date"], x["time"], x["destination"]))
    return jsonify(schedule)


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.json or request.form.to_dict()

    try:
        row = build_prediction_row(form_data)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400

    proba = float(_model.predict_proba(row)[0][1])
    return jsonify(
        {
            "probability": round(proba * 100, 1),
            "is_delayed": proba >= 0.5,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
