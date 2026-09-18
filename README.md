# ProcessIQ — Control Tower Dashboard

A local web dashboard for the ProcessIQ maturity model: data-governance scoring,
maturity leveling, Measured/Optimized readiness gates, rule-based gap detection, and a
RandomForest classifier that predicts maturity level from process signals —
all from your original scoring notebook, wrapped in an interactive UI.

## What's inside

```
processiq-dashboard/
├── app.py                  # Flask app + API endpoints
├── processiq_engine.py     # Ported governance/maturity/gates/rules logic
├── generate_data.py        # Creates a synthetic sample dataset (50 processes)
├── requirements.txt
├── data/
│   └── processiq_v2_corrected_dataset.csv   # sample data (replace with your real export)
├── model/
│   └── processiq_v2_maturity_model.pkl      # trained model (regenerated on start)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/dashboard.js
```

## Run it (VS Code / local machine)

1. Open this folder in VS Code.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python app.py
   ```

4. Open **http://127.0.0.1:5000** in your browser.

On startup the app loads `data/processiq_v2_corrected_dataset.csv`, runs the
full scoring pipeline, and trains the RandomForest maturity-level model —
exactly the same steps as the original notebook, just served as JSON instead
of printed to console.

## Deploy to Render

Create a Render Web Service using the folder that contains `app.py` as the
service root. Use these settings:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`

Render supplies the `PORT` environment variable automatically. The Flask app
binds to `0.0.0.0` and uses that port, so the dashboard and CSV upload flow are
available from the public Render URL.

## Using your real dataset

The bundled CSV is **synthetic sample data** (50 fictional processes) so the
dashboard works immediately. To use your real data, either:

- Replace `data/processiq_v2_corrected_dataset.csv` with your export and
  restart the app, or
- Use the **Upload CSV** button in the sidebar to swap datasets live —
  no restart needed. The model retrains automatically on upload.

Your CSV needs the same columns the original notebook expects: the 8 data
governance dimensions, the 5 maturity sub-scores, the gate-related fields
(`event_data_available`, `execution_data_quality`, `process_adherence`,
`deviation_monitoring`, `ai_data_readiness`), and the 34 features used by the
classifier. If your CSV already contains `all_gaps`, `triggered_rules`, and
`recommendations` columns those are used as-is; otherwise the app derives
them live from the same rule thresholds as the original `RULE_DESCRIPTIONS`.

## Dashboard views

- **Overview** — fleet-wide KPI strip, maturity-level distribution, governance
  status breakdown, governance-dimension radar, maturity by department, and
  the most frequently triggered rules across the fleet.
- **Process Registry** — searchable/filterable table of every process; click
  a row to open a full control-tower readout (governance breakdown, gates,
  performance indicators, gaps, triggered rules, recommendations, and target
  next state) — the same content the notebook's `processiq_output()` printed.
- **Model Insights** — RandomForest accuracy, per-class precision/recall/F1,
  and a feature-importance chart.

## Notes

- Regenerate the sample dataset any time with `python generate_data.py`.
- The API is plain JSON (`/api/overview`, `/api/processes`,
  `/api/processes/<id>`, `/api/ml`, `/api/upload`, `/api/reset`) if you want
  to wire up a different frontend later.
