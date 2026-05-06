# NYC Real Estate Intelligence

Badri Narayanan Rajendran · CWID 20030350
Stevens Institute of Technology

---

## Overview

This project classifies NYC properties as **undervalued**, **fairly valued**, or **overvalued** using real
data pulled from five public APIs. On top of the classifier I added a hybrid recommender, geospatial
clustering, and a conversational agent that uses Claude to answer questions in plain English.

I built the labeling scheme around a peer-group z-score (no ground-truth valuations needed), trained
four classifiers on identical preprocessing, and wrapped everything in a Streamlit dashboard.

---

## What it does

- Pulls live sales from the NYC Department of Finance (Socrata API)
- Enriches every property with real Census income, NYPD crime, NYC DOE school, and Walk Score data
- Trains kNN, CART, Random Forest, and XGBoost on the same preprocessor and compares them fairly
- Recommends properties using a 0.6 × content + 0.4 × collaborative hybrid score
- Clusters neighborhoods into 8 auto-named archetypes using k-Means
- Exposes everything through a Streamlit app with a Claude-powered chat assistant

---

## Final results

| Model | Accuracy | F1-Macro | ROC-AUC | Cohen's κ |
|---|---|---|---|---|
| **XGBoost ★** | **0.617** | **0.499** | **0.691** | **0.239** |
| Random Forest | 0.617 | 0.482 | 0.684 | 0.223 |
| kNN | 0.562 | 0.458 | 0.621 | 0.169 |
| CART | 0.557 | 0.461 | 0.605 | 0.171 |

XGBoost is the production model. Full pipeline runs end-to-end in **215 seconds** on my MacBook.

---

## Data sources

| Source | What it provides | Coverage |
|---|---|---|
| NYC DOF (Socrata `usep-8jbt`) | Live property sales — price, address, sqft, year, class | 21,272 records |
| US Census ACS5 | Median household income by zip | 100% of zips |
| NYPD Complaint Data (`qgea-i56i` / `5uac-w243`) | Crime counts mapped via precinct → zip | 79% of zips |
| NYC DOE School Locations (`s3k6-pzi2`) | School density per zip (quality proxy) | 119 zips |
| Walk Score API | Walk, transit, bike scores per property | ~5,000 properties (free tier) |

When an API fails or hits a quota, the row falls back to a borough-calibrated value and gets tagged
`source: synthetic_fallback` so the data lineage stays traceable.

---

## How to run

### 1. Setup

```bash
unzip real_estate_ml.zip
cd real_estate_ml

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API keys

Copy the template and add your keys (free tiers work fine):

```bash
cp config/.env.example config/.env
# Edit config/.env with your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   WALKSCORE_API_KEY=...
#   CENSUS_API_KEY=...
#   SOCRATA_APP_TOKEN=...
```

If you skip a key, the pipeline still runs — it just uses synthetic fallback for that source.

### 3. Run the full pipeline

```bash
python main.py --step all
```

This runs ingest → enrich → features → train → recommend → cluster in about 4 minutes
(faster on subsequent runs because of caching).

### 4. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the URL it prints. The dashboard has four pages:
1. **Property Valuator** — enter property details, get a verdict from XGBoost
2. **Smart Recommender** — set budget and priorities, get ranked matches with a map
3. **Neighborhood Map** — Folium map with cluster overlay and four heatmaps
4. **AI Assistant** — chat with Claude; the seven tools fire in the sidebar in real time

### 5. Other useful commands

```bash
# Run only one stage
python main.py --step train         # retrain classifiers only
python main.py --step recommend     # rebuild recommenders only
python main.py --step cluster       # rebuild clustering only

# Run the test suite (30 tests)
pytest tests/ -v

# Open the experiment tracker
mlflow ui --backend-store-uri ./mlruns
```

---

## Project layout

```
real_estate_ml/
├── README.md
├── SETUP_GUIDE.md
├── requirements.txt
├── main.py                          # CLI entry point
├── config/
│   ├── config.yaml                  # pipeline config
│   └── .env.example                 # API key template
├── src/
│   ├── data/
│   │   ├── ingestion.py             # orchestrates real + synthetic
│   │   ├── socrata_client.py        # NYC Open Data SODA client
│   │   ├── live_integration.py      # merges live DOF rows with enrichers
│   │   ├── hyperlocal_enricher.py   # Walk Score, Census, NYPD, DOE
│   │   ├── synthetic.py             # NYC-calibrated synthetic generator
│   │   ├── enrichment.py            # neighborhood percentiles
│   │   └── validation.py            # pandera schemas
│   ├── features/
│   │   ├── engineering.py           # peer-group z-score labeling
│   │   └── preprocessor.py          # shared ColumnTransformer
│   ├── models/
│   │   ├── classification/          # kNN, CART, RF, XGBoost + base
│   │   ├── recommendation/          # content + collab SVD + hybrid
│   │   └── clustering/              # k-Means + hierarchical
│   ├── evaluation/                  # uniform metrics
│   ├── visualization/               # EDA, model, and geo plots
│   ├── agent/
│   │   ├── tools.py                 # 7 ML-backed tools
│   │   ├── prompts.py
│   │   └── llm_agent.py             # native Anthropic tool-use loop
│   └── utils/                       # config, logger, I/O
├── app/
│   ├── streamlit_app.py
│   ├── components/                  # theme + reusable UI
│   └── pages/                       # 4 pages
├── notebooks/                       # 01–07 (EDA, modeling, evaluation)
├── tests/                           # 30 unit tests
├── models/                          # trained .pkl files
├── data/
│   ├── raw/                         # ingested
│   ├── interim/                     # enriched
│   ├── processed/                   # features + clustered
│   └── external/hyperlocal_cache/   # 7-day API response cache
└── reports/
    ├── classifier_comparison.csv
    └── figures/
```

---

## Methodology highlights

**Peer-group z-score labeling.** The label is derived from how a property's price-per-sqft compares
to the median and standard deviation of its neighborhood peers. `z < −0.75` → undervalued,
`z > 0.75` → overvalued, otherwise fairly valued. No ground-truth labels needed.

**Leakage-safe preprocessor.** All four classifiers share the same `ColumnTransformer` (StandardScaler
for numerics, OrdinalEncoder for categoricals). `price_per_sqft` is excluded from features so the
label can't leak back in.

**Tuned with RandomizedSearchCV.** Each model gets a different search budget (10 iters for kNN/XGBoost,
8 for RF, 15 for CART), 3-fold stratified CV, F1-macro scoring. Same train-test split for all.

**Hybrid recommender with graceful degradation.** When `scikit-surprise` can't be imported (NumPy 2.x
ABI mismatch is common), the collaborative side falls back to user-mean. Documented in code.

**Honest data tagging.** Every output row carries a `data_source` column. Every API response carries
a `source` field — `walkscore_api`, `census_api`, `nyc_open_data_complaints`,
`nyc_open_data_doe_directory`, or `synthetic_fallback`. No hidden fabrication.

---

## AI Disclosure

I used Anthropic Claude in two ways during this project:

1. **As a coding assistant** for debugging API integration errors, suggesting fixes for library
   compatibility issues (NumPy 2.x with scikit-surprise), and reviewing code structure.
2. **As the LLM backend** for the conversational agent in the Streamlit dashboard.

All architectural decisions, the labeling methodology, the choice of features, the evaluation
strategy, and the project scope are mine. I read and tested every suggestion before incorporating
it. See `reports/ai_prompts_used.md` for the full prompt log.

---

## Known limitations

- Walk Score free tier caps at 5,000 calls/day. Properties beyond that get borough-calibrated
  fallbacks. The circuit breaker in `hyperlocal_enricher.py` handles this.
- Crime rate uses a flat 50,000-resident assumption per zip, which introduces error of ±50%
  for very small or very large zips.
- School quality is a density proxy (count of schools in zip), not a measured outcome metric.
- 37 of 173 NYC zips don't have a precinct mapping in my lookup table and use borough fallback
  for crime data.

These are documented in `src/data/hyperlocal_enricher.py` and the project presentation.

---