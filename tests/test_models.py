# =============================================================================
# Course   : CS 513 - Data Analytics & Machine Learning
# Purpose  : Tests for classifier + recommender + clustering modules
# =============================================================================

import pytest

from src.data.enrichment import PropertyEnricher
from src.data.synthetic import generate_synthetic_dataset
from src.evaluation.classifier_metrics import ClassifierEvaluator
from src.features.engineering import FeatureEngineer
from src.features.preprocessor import get_X_y
from src.models.classification.cart import CARTValuationClassifier
from src.models.classification.knn import KNNValuationClassifier
from src.models.clustering.kmeans import NeighborhoodKMeans
from src.models.recommendation.collaborative import (
    CollaborativeRecommender,
    simulate_buyer_interactions,
)
from src.models.recommendation.content_based import ContentBasedRecommender
from src.models.recommendation.hybrid import HybridRecommender
from src.utils.config import load_config


@pytest.fixture(scope="module")
def small_features():
    df = generate_synthetic_dataset(n_samples=2000, random_seed=42)
    df = PropertyEnricher().enrich_all(df)
    return FeatureEngineer().run(df)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# -----------------------------------------------------------------------------
# Classifiers
# -----------------------------------------------------------------------------
def test_cart_can_train_and_predict(small_features, cfg):
    X, y = get_X_y(small_features)
    clf = CARTValuationClassifier(cfg)
    clf.build_pipeline(X)
    clf.train(X, y, tune=False)   # no tuning, just fit
    preds = clf.predict(X.head(50))
    assert len(preds) == 50
    proba = clf.predict_proba(X.head(50))
    assert proba.shape == (50, 3)


def test_knn_can_train(small_features, cfg):
    X, y = get_X_y(small_features)
    clf = KNNValuationClassifier(cfg)
    clf.build_pipeline(X)
    clf.train(X, y, tune=False)
    assert clf.predict(X.head(20)).shape == (20,)


def test_evaluator_returns_complete_metrics(small_features, cfg):
    X, y = get_X_y(small_features)
    clf = CARTValuationClassifier(cfg)
    clf.build_pipeline(X)
    clf.train(X, y, tune=False)

    ev = ClassifierEvaluator()
    res = ev.evaluate(clf, X, y, train_seconds=1.0, name="cart")
    assert 0 <= res.accuracy <= 1
    assert 0 <= res.f1_macro <= 1
    assert res.confusion_matrix.shape == (3, 3)
    assert "support" in res.classification_report or "precision" in res.classification_report


# -----------------------------------------------------------------------------
# Recommenders
# -----------------------------------------------------------------------------
def test_content_recommender_basic(small_features):
    rec = ContentBasedRecommender().fit(small_features)
    out = rec.recommend_similar(0, top_n=5)
    assert len(out) <= 5
    assert "similarity" in out.columns


def test_content_preferences_filter(small_features):
    rec = ContentBasedRecommender().fit(small_features)
    out = rec.recommend_from_preferences(
        {"budget_max": 800_000, "prefer_safe": True}, top_n=5
    )
    assert (out["price"] <= 800_000).all() if not out.empty else True


def test_collaborative_with_fallback(small_features):
    interactions = simulate_buyer_interactions(small_features, n_buyers=20,
                                                  interactions_per_buyer=10)
    assert len(interactions) > 0
    rec = CollaborativeRecommender(n_factors=10, n_epochs=3).fit(interactions)
    rating = rec.predict_rating("buyer_0001",
                                  small_features.iloc[0]["property_id"])
    assert 1 <= rating <= 5


def test_hybrid_blends(small_features):
    interactions = simulate_buyer_interactions(small_features, n_buyers=20,
                                                  interactions_per_buyer=10)
    content = ContentBasedRecommender().fit(small_features)
    collab = CollaborativeRecommender(n_factors=10, n_epochs=3).fit(interactions)
    hybrid = HybridRecommender(content, collab, alpha=0.6)
    out = hybrid.recommend(buyer_id="buyer_0001", seed_property_idx=0, top_n=5)
    if not out.empty:
        assert "hybrid_score" in out.columns


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
def test_kmeans_assigns_clusters(small_features, cfg):
    km = NeighborhoodKMeans(n_clusters=4, random_state=42, n_init=3)
    km.fit(small_features, cfg["clustering"]["features"])
    labels = km.predict(small_features)
    assert len(labels) == len(small_features)
    assert set(labels).issubset(set(range(4)))


def test_kmeans_auto_names(small_features, cfg):
    km = NeighborhoodKMeans(n_clusters=4, random_state=42, n_init=3)
    km.fit(small_features, cfg["clustering"]["features"])
    assert len(km.cluster_names_) == 4
    for name in km.cluster_names_.values():
        assert isinstance(name, str) and len(name) > 0
