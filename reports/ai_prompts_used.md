# AI Usage Disclosure

Badri Narayanan Rajendran · CWID 20030350

---

## Overview

I used Anthropic Claude in two distinct ways during this project:

1. **As a coding assistant** during development (debugging, code review, library
   compatibility fixes, drafting boilerplate).
2. **As the LLM backend** for the conversational agent inside the Streamlit app.

All architectural decisions, the labeling methodology, the model selection, the
evaluation strategy, and the project scope are mine. I read, tested, and verified
every suggestion before incorporating it.

---

## 1. Code Generation Assistance

### Project scaffolding

> "I'm building a CS 513A final project: a hyperlocal NYC real estate
> classification and hybrid recommendation system. Generate a modular Python
> project structure with separation of concerns (src/, tests/, app/, notebooks/),
> proper config management with YAML + .env, and OOP-based classifiers
> inheriting from a common base class."

### Synthetic data generator

> "Generate a Python module that produces realistic synthetic NYC real-estate
> data: 5 boroughs with calibrated price-per-sqft, walk score, crime rate,
> and school quality distributions reflecting actual 2023-2024 averages.
> Ensure feature correlations are economically realistic (price positively
> correlates with walk/schools, negatively with crime)."

### Target labeling methodology

> "Design a defensible methodology for creating valuation labels
> (undervalued / fairly_valued / overvalued) without ground-truth labels.
> Use peer-group statistical comparison via z-scores."

### Real API integration (largest assistance area)

> "I need to replace synthetic hyperlocal features with real values from
> Walk Score API, US Census ACS5, NYPD Complaint Data, and NYC DOE School
> Locations. Help me build a hyperlocal_enricher.py module that calls each
> API, caches responses to disk, and falls back to borough-calibrated
> synthetic values when the API fails or quotas are exceeded."

This required several iterations to fix dataset IDs and field names that I
initially had wrong:

- **NYPD Complaint Data:** I tried `kmqf-h5zq` (which didn't exist) and `5uac-w243`
  with field `zip_cd` (which doesn't exist in that dataset). Eventually settled
  on aggregating `qgea-i56i` and `5uac-w243` by `addr_pct_cd` (precinct), then
  mapping precincts to zips through a manually-compiled lookup table.
- **NYC DOE School Quality:** Dataset `9qzq-t3g5` doesn't exist. I tried
  `dnpx-dfnc` with multiple field-name variants and school-year formats — none
  worked reliably from the SODA API. Final approach: use the school directory
  (`s3k6-pzi2`) alone and derive a quality proxy from school count per zip.
- **Walk Score:** Free-tier quota is 5,000 calls/day. Implemented a circuit
  breaker that disables the API after 10 consecutive failures and falls back
  to borough-calibrated synthetic for remaining properties.

> "The pipeline takes 90 minutes because we make 173 separate API calls for
> Census, crime, and school data. Convert this to a bulk-prefetch pattern
> that fetches all NYC schools and all NYPD complaints in one bulk SoQL query
> each, then maps to zip codes from in-memory dictionaries."

This optimization reduced enrichment runtime from ~30 minutes to ~3 seconds.

### Library compatibility — scikit-surprise + NumPy 2.x

> "scikit-surprise was compiled against NumPy 1.x and crashes on import with
> NumPy 2.4.4. Update the _try_surprise method in collaborative.py to catch
> ImportError, RuntimeError, ValueError, and OSError, then fall back gracefully
> to the user-mean baseline. The pipeline should never fail because of this."

### Streamlit dashboard

> "Build a Streamlit dashboard with four pages: Property Valuator, Smart
> Recommender, Neighborhood Map (Folium with 4 heatmap layers), and AI
> Assistant. Use a clean Material Design aesthetic with cards, animated
> counters, and a data-freshness badge that shows when the live feed was
> last updated."

### Code review and refactoring

> "Review hyperlocal_enricher.py and tell me which fields are real API values
> versus synthetic fallbacks versus derived metrics. I want to be honest in
> my report about which is which."

The result of this review is documented in the README's "Known limitations"
section and in the comments of `hyperlocal_enricher.py`.

### Project documentation

> "Rewrite my README to read like a graduate student wrote it — factual,
> first-person, no marketing language. Include real metrics from my final
> pipeline run, accurate data source coverage, and an honest known-limitations
> section."

### Final presentation

I used Claude to generate the 10-slide PowerPoint deck (`.pptx`) using the
`pptxgenjs` library. The deck content (metrics, methodology, data sources,
and project structure) was supplied by me and reflects the actual project state.

---

## 2. LLM as Agent Backend

The conversational agent in `src/agent/llm_agent.py` uses Claude
(`claude-sonnet-4-6`, with `claude-haiku-4-5` as fallback) via Anthropic's
native tool-use API. The agent has access to **7 ML-backed tools** defined
in `src/agent/tools.py`.

### Tools

| Tool | Backed by |
|---|---|
| `classify_property` | Trained XGBoost classifier |
| `get_recommendations` | Hybrid recommender (content cosine + collaborative SVD) |
| `get_neighborhood_profile` | DataFrame aggregation over the property dataset |
| `compare_properties` | DataFrame lookup |
| `get_market_summary` | DataFrame aggregation with label distribution |
| `get_live_market_data` | NYC DOF live sales feed via Socrata SODA API |
| `check_data_freshness` | Socrata metadata endpoint (dataset update timestamp) |

### System prompt behavior

The system prompt in `src/agent/prompts.py` instructs the agent to:

1. Always ground answers in tool calls (never fabricate numbers).
2. Use live tools when the user asks about recent or current activity.
3. Use historical tools for valuation, recommendations, and comparisons.
4. Mention the data source in its reply when using live data.
5. Gracefully tell the user "live data unreachable, using historical" when
   the API is unavailable.
6. Avoid offering legal or financial advice.

### No LangChain, no RAG, no vector DB

The agent is a simple multi-turn loop using Anthropic's native `tool_use` API.
Tool definitions are JSON-schema. The loop runs up to 6 turns per user message.

---

## 3. Reproducibility

- All randomness is seeded via `config.yaml::project.random_seed = 42`.
- The synthetic data generator is deterministic.
- API responses are cached in `data/external/hyperlocal_cache/` with a 7-day TTL.
- The full pipeline can be re-run end-to-end: `python main.py --step all`.
- Tests: 30 unit tests, all passing on a clean checkout (`pytest tests/ -v`).
- Final pipeline runtime: 215 seconds on my MacBook (Apple Silicon).

---

## 4. What was mine, not Claude's

To be clear about what required my own judgment:

- The decision to use a peer-group z-score for labeling instead of trying
  to obtain ground-truth valuations.
- The choice to compare four classifiers (kNN, CART, RF, XGBoost) on identical
  preprocessing rather than tuning each in isolation.
- The decision to exclude `price_per_sqft` from the feature matrix to prevent
  label leakage.
- The decision to use a hybrid recommender weighted 0.6 / 0.4 between content
  and collaborative.
- The decision to map NYPD precincts to zip codes manually rather than chase
  a non-existent zip-level NYPD dataset.
- The decision to keep synthetic fallback rather than fail loudly when an
  API is unavailable, and to tag every row with its data source.
- The decision to acknowledge limitations honestly in the README rather than
  oversell the system.

I read every code suggestion, ran the tests, and rejected several proposals
that would have introduced data leakage or unrealistic assumptions.