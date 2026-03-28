"""Build a static HTML site with all predictions pre-computed.

Outputs docs/index.html with all data embedded as JSON — no backend needed.
"""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"


def _generate_time(seed_str, index):
    h = int(hashlib.md5(f"{seed_str}-{index}".encode()).hexdigest()[:8], 16)
    hour = 6 + (h % 16)
    minute = (h >> 4) % 4 * 15
    return f"{hour:02d}:{minute:02d}"


def _generate_flight_number(airline, dest, index):
    code = airline[:2].upper().replace(" ", "")
    h = int(hashlib.md5(f"{airline}-{dest}-{index}".encode()).hexdigest()[:6], 16)
    num = 100 + (h % 900)
    return f"{code}{num}"


def build_prediction_row(form_data, columns, route_stats, numeric_defaults):
    row = pd.DataFrame(0, index=[0], columns=columns)
    airport = form_data["reporting_airport"]
    country = form_data["origin_destination_country"]
    stats = route_stats.get((airport, country), numeric_defaults)
    for feat in NUMERIC_FEATURES:
        row[feat] = stats[feat]
    for feat in CATEGORICAL_FEATURES:
        val = form_data[feat]
        col_name = f"{feat}_{val}"
        if col_name in columns:
            row[col_name] = 1
    return row


def build():
    model = joblib.load(MODELS_DIR / "model.joblib")
    meta = joblib.load(MODELS_DIR / "model_metadata.joblib")
    columns = meta["columns"]
    category_values = meta["category_values"]
    numeric_defaults = meta["numeric_defaults"]
    route_stats = meta["route_stats"]
    route_airlines = meta["route_airlines"]
    flights = meta["flights"]

    # Derived lookups
    route_airlines_js = {f"{k[0]}||{k[1]}": v for k, v in route_airlines.items()}
    airport_countries = {}
    for airport, country in route_airlines:
        airport_countries.setdefault(airport, []).append(country)
    airport_countries = {k: sorted(set(v)) for k, v in airport_countries.items()}

    # Pre-compute predictions for every (airport, country, airline, direction) combo
    predictions = {}
    for key_str, dests in flights.items():
        parts = key_str.split("||")
        airport, country, airline, direction = parts
        form_data = {
            "reporting_airport": airport,
            "origin_destination_country": country,
            "airline_name": airline,
            "arrival_departure": direction,
        }
        row = build_prediction_row(form_data, columns, route_stats, numeric_defaults)
        proba = float(model.predict_proba(row)[0][1])
        predictions[key_str] = round(proba * 100, 1)

    # Pre-compute flight schedules
    today = date.today()
    all_flights = {}
    for key_str, dests in flights.items():
        parts = key_str.split("||")
        airport, country, airline, direction = parts
        schedule = []
        for dest_info in dests:
            dest = dest_info["destination"]
            monthly = dest_info["flights_per_month"]
            daily = max(1, round(monthly / 30))
            for day_offset in range(7):
                flight_date = today + timedelta(days=day_offset)
                for i in range(daily):
                    seed = f"{airport}-{dest}-{airline}-{direction}-{i}"
                    schedule.append({
                        "flight_number": _generate_flight_number(airline, dest, i),
                        "destination": dest,
                        "date": flight_date.isoformat(),
                        "time": _generate_time(seed, 0),
                        "avg_delay": dest_info["avg_delay"],
                    })
        schedule.sort(key=lambda x: (x["date"], x["time"], x["destination"]))
        all_flights[key_str] = schedule

    # Read the HTML template and replace Jinja with baked data
    template = (ROOT / "templates" / "index.html").read_text()

    # Build static HTML
    airports_json = json.dumps(category_values["reporting_airport"])
    airport_countries_json = json.dumps(airport_countries)
    route_airlines_json = json.dumps(route_airlines_js)
    predictions_json = json.dumps(predictions)
    flights_json = json.dumps(all_flights)

    html = _build_static_html(
        airports_json, airport_countries_json, route_airlines_json,
        predictions_json, flights_json,
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html)
    print(f"Built static site: {DOCS_DIR / 'index.html'}")
    print(f"  Routes: {len(predictions)}")
    print(f"  Flights: {sum(len(v) for v in all_flights.values())}")


def _build_static_html(airports, airport_countries, route_airlines, predictions, flights):
    """Generate the full static HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Flight Delay Predictor</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0c1222; color: #e2e8f0; min-height: 100vh;
    }}
    .app {{ max-width: 640px; margin: 0 auto; padding: 2.5rem 1.25rem 3rem; }}
    .header {{ text-align: center; margin-bottom: 2rem; }}
    .header h1 {{ font-size: 1.75rem; font-weight: 800; color: #f8fafc; }}
    .header p {{ font-size: 0.85rem; color: #64748b; margin-top: 0.25rem; }}
    .step {{ margin-bottom: 1.5rem; }}
    .step-label {{
      font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; color: #3b82f6; margin-bottom: 0.5rem;
    }}
    .card {{
      background: rgba(30, 41, 59, 0.7); border: 1px solid #334155;
      border-radius: 14px; padding: 1.25rem 1.5rem;
    }}
    label {{
      display: block; font-size: 0.8rem; font-weight: 600;
      color: #94a3b8; margin-bottom: 0.35rem;
    }}
    select {{
      width: 100%; padding: 0.6rem 2rem 0.6rem 0.75rem;
      border: 1px solid #334155; border-radius: 10px;
      background: #0f172a; color: #e2e8f0; font-size: 0.9rem;
      appearance: none; cursor: pointer; transition: border-color 0.2s;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%2364748b' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 0.75rem center;
    }}
    select:focus {{ outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }}
    select:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .row {{ display: flex; gap: 0.75rem; }}
    .row > * {{ flex: 1; }}
    .toggle-row {{ display: flex; gap: 0; border-radius: 10px; overflow: hidden; border: 1px solid #334155; }}
    .toggle-row input {{ display: none; }}
    .toggle-row label {{
      flex: 1; text-align: center; padding: 0.55rem 0; font-size: 0.85rem;
      font-weight: 600; color: #64748b; background: #0f172a; cursor: pointer;
      margin: 0; transition: all 0.2s; user-select: none;
    }}
    .toggle-row input:checked + label {{ background: #1e40af; color: #fff; }}
    .flights-section {{ margin-top: 1.5rem; }}
    .flights-section .step-label {{ margin-bottom: 0.75rem; }}
    .date-group-label {{
      font-size: 0.75rem; font-weight: 700; color: #64748b;
      padding: 0.6rem 0 0.3rem; border-bottom: 1px solid #1e293b;
      margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .date-group-label:first-child {{ padding-top: 0; }}
    .flight-time {{
      font-size: 1.1rem; font-weight: 700; color: #f8fafc;
      min-width: 50px; font-variant-numeric: tabular-nums;
    }}
    .flight-card {{
      display: flex; align-items: center; gap: 1rem;
      padding: 0.75rem 1rem; border-radius: 10px;
      border: 1px solid transparent; cursor: pointer;
      transition: all 0.15s; margin-bottom: 0.35rem;
    }}
    .flight-card:hover {{ background: rgba(59,130,246,0.06); border-color: #334155; }}
    .flight-card.selected {{ background: rgba(59,130,246,0.1); border-color: #3b82f6; }}
    .flight-info {{ flex: 1; min-width: 0; }}
    .flight-dest {{
      font-size: 0.9rem; font-weight: 600; color: #cbd5e1;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .flight-num {{ font-size: 0.75rem; color: #64748b; }}
    .flight-arrow {{
      width: 28px; height: 28px; border-radius: 50%;
      background: #1e293b; display: flex; align-items: center;
      justify-content: center; flex-shrink: 0; transition: background 0.15s;
    }}
    .flight-card.selected .flight-arrow {{ background: #3b82f6; }}
    .flight-arrow svg {{ width: 14px; height: 14px; fill: #94a3b8; }}
    .flight-card.selected .flight-arrow svg {{ fill: #fff; }}
    .flights-empty {{ text-align: center; padding: 2rem 0; color: #475569; font-size: 0.9rem; }}
    .flights-scroll {{ max-height: 420px; overflow-y: auto; padding-right: 0.25rem; }}
    .flights-scroll::-webkit-scrollbar {{ width: 4px; }}
    .flights-scroll::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 2px; }}
    .result-card {{
      text-align: center; padding: 2rem 1.5rem;
      background: rgba(30, 41, 59, 0.7); border: 1px solid #334155;
      border-radius: 14px; margin-top: 1.5rem;
      animation: fadeUp 0.4s ease;
    }}
    @keyframes fadeUp {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .result-prob {{ font-size: 3.5rem; font-weight: 800; line-height: 1; margin-bottom: 0.5rem; }}
    .result-badge {{
      display: inline-block; font-size: 1rem; font-weight: 700;
      padding: 0.35rem 1.1rem; border-radius: 999px;
    }}
    .result-card.delayed .result-prob {{ color: #f87171; }}
    .result-card.delayed .result-badge {{ background: rgba(239,68,68,0.12); color: #f87171; }}
    .result-card.ontime .result-prob {{ color: #4ade80; }}
    .result-card.ontime .result-badge {{ background: rgba(34,197,94,0.12); color: #4ade80; }}
    .result-delay {{ margin-top: 0.75rem; font-size: 0.95rem; color: #94a3b8; }}
    .result-delay strong {{ color: #e2e8f0; }}
    .gauge {{ margin: 1.25rem auto 0; max-width: 280px; }}
    .gauge-track {{ height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; }}
    .gauge-fill {{
      height: 100%; border-radius: 4px;
      background: linear-gradient(90deg, #4ade80, #facc15 50%, #f87171);
      transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    .gauge-labels {{
      display: flex; justify-content: space-between;
      font-size: 0.65rem; color: #475569; margin-top: 0.3rem;
    }}
    footer {{ text-align: center; margin-top: 3rem; font-size: 0.7rem; color: #334155; }}
  </style>
</head>
<body>
<div class="app">
  <div class="header">
    <h1>Flight Delay Predictor</h1>
    <p>UK CAA Punctuality Data</p>
  </div>
  <div class="step">
    <div class="step-label">1 &mdash; Choose your route</div>
    <div class="card">
      <div style="margin-bottom:1rem">
        <label for="airport">Airport</label>
        <select id="airport"></select>
      </div>
      <div class="row" style="margin-bottom:1rem">
        <div>
          <label for="country">Country</label>
          <select id="country" disabled><option>Select airport</option></select>
        </div>
        <div>
          <label for="airline">Airline</label>
          <select id="airline" disabled><option>Select route</option></select>
        </div>
      </div>
      <div>
        <label>Direction</label>
        <div class="toggle-row">
          <input type="radio" name="direction" id="dep" value="D" checked>
          <label for="dep">Departure</label>
          <input type="radio" name="direction" id="arr" value="A">
          <label for="arr">Arrival</label>
        </div>
      </div>
    </div>
  </div>
  <div class="flights-section" id="flights-section" style="display:none">
    <div class="step-label">2 &mdash; Pick a flight</div>
    <div class="card">
      <div id="flights-container" class="flights-scroll"></div>
    </div>
  </div>
  <div id="result-section"></div>
  <footer>Random Forest baseline &middot; scikit-learn</footer>
</div>
<script>
const airports = {airports};
const airportCountries = {airport_countries};
const routeAirlines = {route_airlines};
const allFlights = {flights};
const predictions = {predictions};

const airportEl = document.getElementById('airport');
const countryEl = document.getElementById('country');
const airlineEl = document.getElementById('airline');
const flightsSection = document.getElementById('flights-section');
const flightsContainer = document.getElementById('flights-container');
const resultSection = document.getElementById('result-section');

// Populate airports
airports.forEach(a => {{
  const opt = document.createElement('option');
  opt.value = a; opt.textContent = a;
  airportEl.appendChild(opt);
}});

function getDirection() {{
  return document.querySelector('input[name="direction"]:checked').value;
}}

function populateSelect(el, items, placeholder) {{
  el.innerHTML = '';
  if (!items || items.length === 0) {{
    el.innerHTML = '<option value="">' + (placeholder || 'None available') + '</option>';
    el.disabled = true; return;
  }}
  items.forEach(v => {{
    const opt = document.createElement('option');
    opt.value = v; opt.textContent = v;
    el.appendChild(opt);
  }});
  el.disabled = false;
}}

function updateCountries() {{
  const countries = airportCountries[airportEl.value] || [];
  populateSelect(countryEl, countries, 'No routes');
  updateAirlines();
}}

function updateAirlines() {{
  const key = airportEl.value + '||' + countryEl.value;
  const airlines = routeAirlines[key] || [];
  populateSelect(airlineEl, airlines, 'No airlines');
  loadFlights();
}}

function getRouteKey() {{
  return airportEl.value + '||' + countryEl.value + '||' + airlineEl.value + '||' + getDirection();
}}

function loadFlights() {{
  const airport = airportEl.value;
  const country = countryEl.value;
  const airline = airlineEl.value;
  if (!airport || !country || !airline) {{
    flightsSection.style.display = 'none';
    resultSection.innerHTML = '';
    return;
  }}
  const key = getRouteKey();
  const flights = allFlights[key] || [];
  flightsSection.style.display = 'block';
  resultSection.innerHTML = '';
  renderFlights(flights);
}}

function renderFlights(flights) {{
  if (!flights.length) {{
    flightsContainer.innerHTML = '<div class="flights-empty">No flights found</div>';
    return;
  }}
  let html = '';
  let currentDate = '';
  flights.forEach((f, i) => {{
    if (f.date !== currentDate) {{
      currentDate = f.date;
      const d = new Date(f.date + 'T00:00:00');
      const label = d.toLocaleDateString('en-GB', {{ weekday: 'short', day: 'numeric', month: 'short' }});
      html += '<div class="date-group-label">' + label + '</div>';
    }}
    html += '<div class="flight-card" data-index="' + i + '"'
      + ' data-dest="' + f.destination + '"'
      + ' data-flight="' + f.flight_number + '"'
      + ' data-avg-delay="' + (f.avg_delay != null ? f.avg_delay : '') + '">'
      + '<div class="flight-time">' + f.time + '</div>'
      + '<div class="flight-info">'
      + '<div class="flight-dest">' + f.destination + '</div>'
      + '<div class="flight-num">' + f.flight_number + '</div>'
      + '</div>'
      + '<div class="flight-arrow"><svg viewBox="0 0 16 16"><path d="M6 3l5 5-5 5"/></svg></div>'
      + '</div>';
  }});
  flightsContainer.innerHTML = html;
  flightsContainer.querySelectorAll('.flight-card').forEach(card => {{
    card.addEventListener('click', () => selectFlight(card));
  }});
}}

function selectFlight(card) {{
  flightsContainer.querySelectorAll('.flight-card').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');
  const avgDelay = card.dataset.avgDelay;
  const key = getRouteKey();
  const probability = predictions[key];
  if (probability === undefined) {{
    resultSection.innerHTML = '<div class="result-card"><p style="color:#f87171">No prediction available</p></div>';
    return;
  }}
  const isDelayed = probability >= 50;
  const cls = isDelayed ? 'delayed' : 'ontime';
  let delayLine = '';
  if (avgDelay && parseFloat(avgDelay) > 0) {{
    const mins = Math.round(parseFloat(avgDelay));
    const delayStr = mins >= 60 ? Math.floor(mins/60) + 'h ' + (mins%60) + 'm' : mins + ' min';
    delayLine = '<div class="result-delay">Avg delay: <strong>' + delayStr + '</strong></div>';
  }}
  resultSection.innerHTML = '<div class="result-card ' + cls + '">'
    + '<div class="result-prob">' + probability + '%</div>'
    + '<div class="result-badge">chance of delay</div>'
    + delayLine
    + '<div class="gauge"><div class="gauge-track">'
    + '<div class="gauge-fill" style="width:' + probability + '%"></div>'
    + '</div><div class="gauge-labels"><span>Low risk</span><span>High risk</span></div></div>'
    + '</div>';
  resultSection.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}

airportEl.addEventListener('change', updateCountries);
countryEl.addEventListener('change', updateAirlines);
airlineEl.addEventListener('change', loadFlights);
document.querySelectorAll('input[name="direction"]').forEach(r =>
  r.addEventListener('change', loadFlights)
);
updateCountries();
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
