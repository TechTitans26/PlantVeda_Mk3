import os
import requests
from flask import Flask, render_template, request, send_file  # add send_file
from datetime import datetime  # add this
import PlantVeda_SuperRoute_Mk3 as super_route

app = Flask(__name__)

REPORT_DIR = os.path.join('static', 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

LOCATIONIQ_KEY = os.getenv("LOCATIONIQ_KEY")

@app.route("/", methods=["GET", "POST"])
def index():
    print("ROUTE HIT")

    if request.method == "POST":
        try:
            # ── 1. Read form inputs ────────────────────────────────────────────
            place  = request.form.get("place", "").strip()
            area   = float(request.form.get("area"))
            budget = float(request.form.get("budget"))
            soil   = request.form.get("soil")
            habitat = request.form.get("habitat")

            if not place:
                return render_template("index.html", error="Enter a location.")

            print(f"Place: {place} | Area: {area} | Budget: {budget} | Soil: {soil} | Habitat: {habitat}")

            # ── 2. Geocoding ───────────────────────────────────────────────────
            geo_url = (
                "https://us1.locationiq.com/v1/search"
                f"?key={LOCATIONIQ_KEY}&q={requests.utils.quote(place)}&format=json"
            )
            geo_res = requests.get(geo_url, timeout=10)
            geo_res.raise_for_status()
            geo_data = geo_res.json()

            if not geo_data:
                return render_template("index.html", error="Place not found.")

            lat = float(geo_data[0]["lat"])
            lon = float(geo_data[0]["lon"])
            print(f"Lat: {lat}  Lon: {lon}")

           # ── 3. Fetch Weather Data (Sync URL with extraction) ──
            # We explicitly ask for 'daily' (for avg temp) and 'hourly' (for sun peak)
            w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min&hourly=shortwave_radiation&timezone=auto&forecast_days=1"
            
            w_response = requests.get(w_url)
            if w_response.status_code != 200:
                return render_template("index.html", error="Weather API communication failed.")
            
            w_json = w_response.json()

            # ── 4. Process Numerical Data Safely ──
            # Use .get() to avoid 'KeyError' crashes
            daily_data = w_json.get('daily', {})
            t_max = daily_data.get('temperature_2m_max', [25])[0] # Default 25 if missing
            t_min = daily_data.get('temperature_2m_min', [15])[0] # Default 15 if missing
            avg_temp = (t_max + t_min) / 2

            hourly_data = w_json.get('hourly', {})
            radiation_list = hourly_data.get('shortwave_radiation', [])
            peak_radiation = max(radiation_list) if radiation_list else 0.0

            # ── 5. Classify into labels ──
            full_sun_hours = sum(1 for r in radiation_list if r > 400)
            sunlight = "Full sun" if full_sun_hours >= 6 else "Partial sun"
            # Format temp_range correctly for the ML models
            temp_range = f"{int(avg_temp)-2}-{int(avg_temp)+2}"

            pdf_buffer = super_route.run_pipeline(
            area, budget, soil, habitat, temp_range, sunlight
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=False,
            download_name=f"PlantVeda_Report_{timestamp}.pdf"
            )
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
