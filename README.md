# Predicting Forest Cover Type from Cartographic Variables

DSC 148 course project. We predict which of 7 tree-cover types dominates a
30×30 m forest patch, using only cheap cartographic variables (elevation,
slope, distances, illumination). Dataset: UCI / scikit-learn **Covertype**,
**581,012** instances, 54 features, 7 classes.

```
.
├── run_all.py        # one command: EDA + baselines + proposed model + ablation + plots
├── app.py            # Streamlit demo (the bonus): interactive predictor
├── features.py       # schema + engineered features (shared by training & demo)
├── requirements.txt
├── report/report.tex # ACM two-column conference report
├── figures/          # created by run_all.py
└── results/          # created by run_all.py (metrics.json, eda_summary.txt)
```

## Quickstart (≈10 min hands-on, ~5 min of waiting)

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run_all.py        # downloads data (~11 MB), trains everything, makes figures
                         # prints a results table you paste into the report
```

`run_all.py` writes `model.joblib`, `figures/*.png`, and `results/metrics.json`.
On a laptop the full run is ~3–6 minutes. To go faster: `python run_all.py --subsample 150000`.

## Run the demo

```bash
streamlit run app.py
```

This opens a browser app, where you can move the sliders for elevation, slope, distances, etc.,
pick a wilderness area and soil type, and the model returns the predicted
cover type plus class probabilities/


## Reproducibility
-> can use `random_state=42`. Re-running gives identical numbers.
