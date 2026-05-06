#!/usr/bin/env python

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

from src.data.synthetic import generate_synthetic_dataset
from src.data.enrichment import PropertyEnricher
from src.features.engineering import FeatureEngineer
from src.features.preprocessor import get_X_y
from src.models.classification.cart import CARTValuationClassifier
from src.evaluation.classifier_metrics import ClassifierEvaluator
from src.models.recommendation.content_based import ContentBasedRecommender
from src.models.clustering.kmeans import NeighborhoodKMeans
from src.utils.config import load_config

print("=" * 60)
print("CS 513 SMOKE TEST")
print("=" * 60)

cfg = load_config()

print("\n[1/6] Generating 3000 synthetic properties...")
df = generate_synthetic_dataset(n_samples=3000, random_seed=42)
print(f"      shape={df.shape}")

print("\n[2/6] Enriching with neighborhood percentiles...")
df = PropertyEnricher().enrich_all(df)
print(f"      shape={df.shape}")

print("\n[3/6] Building features + labels...")
df = FeatureEngineer().run(df)
dist = df["valuation_label_name"].value_counts(normalize=True).round(3)
print(f"      label distribution: {dict(dist)}")

print("\n[4/6] Training CART classifier...")
X, y = get_X_y(df)
clf = CARTValuationClassifier(cfg)
clf.build_pipeline(X)
clf.train(X, y, tune=False)
ev = ClassifierEvaluator()
res = ev.evaluate(clf, X, y, train_seconds=0, name="cart")
print(f"      training accuracy={res.accuracy:.3f}, f1_macro={res.f1_macro:.3f}")

print("\n[5/6] Building content-based recommender...")
rec = ContentBasedRecommender().fit(df)
out = rec.recommend_from_preferences(
    {"budget_max": 1_500_000, "prefer_safe": True, "prefer_school": True}, top_n=3
)
print(f"      top match: {out.iloc[0]['borough']} @ ${out.iloc[0]['price']:,.0f}")

print("\n[6/6] k-Means clustering...")
km = NeighborhoodKMeans(n_clusters=4, random_state=42, n_init=3)
km.fit(df, cfg["clustering"]["features"])
print(f"      cluster archetypes: {list(km.cluster_names_.values())}")

print("\n" + "=" * 60)
print("✅ ALL SMOKE TESTS PASSED")
print("=" * 60)
print("\nReady to run full pipeline:  python main.py --step all")
print("Then launch app:              streamlit run app/streamlit_app.py")
