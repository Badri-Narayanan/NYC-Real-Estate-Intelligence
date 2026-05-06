from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _has_parquet_engine() -> bool:
    """Return True if pyarrow or fastparquet is importable."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame. Uses parquet if engine available, else falls back to pickle.

    The function name is preserved for API stability; an environment without
    pyarrow will write a .pkl with the *same stem* and log the substitution.
    """
    p = Path(path)
    ensure_dir(p.parent)
    if _has_parquet_engine():
        df.to_parquet(p, index=False, compression="snappy")
        log.info(f"Saved {len(df):,} rows (parquet) -> {p}")
    else:
        pkl = p.with_suffix(".pkl")
        df.to_pickle(pkl)
        log.warning(
            f"pyarrow/fastparquet not installed; saved pickle instead -> {pkl}"
        )


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load DataFrame. Tries parquet first, falls back to pickle with same stem."""
    p = Path(path)
    if p.exists() and _has_parquet_engine():
        df = pd.read_parquet(p)
        log.info(f"Loaded {len(df):,} rows (parquet) <- {p}")
        return df
    pkl = p.with_suffix(".pkl")
    if pkl.exists():
        df = pd.read_pickle(pkl)
        log.info(f"Loaded {len(df):,} rows (pickle) <- {pkl}")
        return df
    raise FileNotFoundError(f"Neither {p} nor {pkl} found")


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame as CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    log.info(f"Saved CSV {len(df):,} rows -> {p}")


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save any Python object (model, pipeline) using joblib."""
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)
    log.info(f"Pickled object -> {p}")


def load_pickle(path: str | Path) -> Any:
    """Load any pickled Python object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle file not found: {p}")
    obj = joblib.load(p)
    log.info(f"Loaded pickled object <- {p}")
    return obj
