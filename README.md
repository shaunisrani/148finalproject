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
No internet on your machine? `python run_all.py --smoke` proves the code runs on tiny synthetic data.

## Run the demo (bonus, up to 5%)

```bash
streamlit run app.py
```

Opens a browser app: move the sliders for elevation, slope, distances, etc.,
pick a wilderness area and soil type, and the model returns the predicted
cover type plus class probabilities — exactly the "give your own inputs and
check the prediction" demo the brief asks for.

## Compile the report

Easiest (matches the course's recommended ACM template):
1. Go to Overleaf → New Project → Upload, and upload `report/report.tex`.
   Also upload the `figures/` folder into the project root.
2. Make sure the compiler is **pdfLaTeX**. Click **Recompile**.

The `.tex` already contains realistic results. After your `run_all.py` finishes,
replace the two numbers tables (clearly marked `% <-- REPLACE`) with the table
that the script prints, so the report shows *your* run.

Local alternative: `cd report && latexmk -pdf report.tex` (needs a TeX install with `acmart`).

## Put it on GitHub

```bash
cd <this folder>
git init
git add .
git commit -m "DSC 148 project: forest cover type classification"

# Option A — GitHub CLI:
gh repo create dsc148-covertype --public --source=. --remote=origin --push

# Option B — website: create an empty repo at github.com/new, then:
git remote add origin https://github.com/<you>/dsc148-covertype.git
git branch -M main
git push -u origin main
```

`model.joblib`, `figures/`, and `results/` are git-ignored (regenerate them with
`run_all.py`). Submit the repo URL with your report.

## Reproducibility
All splits and models use `random_state=42`. Re-running gives identical numbers.
