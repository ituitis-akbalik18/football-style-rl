"""
Style Vector Construction & Normalisation
------------------------------------------
Converts raw aggregated team features into normalised style vectors z ∈ [0,1]^d
that can be directly fed into RL policy conditioning.

Two representations are provided:
  1. DirectStyleVector   – interpretable min-max scaled vector
  2. PCAStyleVector      – lower-dimensional learned embedding (k=3–5 dims)

Usage
-----
    from src.style_vectors.style_encoder import StyleEncoder

    encoder = StyleEncoder()
    encoder.fit(team_features_df)             # fit scaler on the dataset
    z = encoder.transform(team_features_df)   # shape: (n_teams, n_features)

    # For a single team by name:
    z_city = encoder.get_vector("Manchester City")

    # Save / load
    encoder.save("data/processed/style_encoder.pkl")
    encoder = StyleEncoder.load("data/processed/style_encoder.pkl")
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from src.features.team_features import FEATURE_COLS, FEATURE_META

logger = logging.getLogger(__name__)


class StyleEncoder:
    """
    Fits a scaler on a team feature DataFrame and produces style vectors.

    Parameters
    ----------
    n_pca_components : int or None
        If set, also computes a lower-dimensional PCA embedding.
    """

    def __init__(self, n_pca_components: Optional[int] = None):
        self.n_pca_components = n_pca_components
        self.scaler = MinMaxScaler(clip=True)
        self.pca: Optional[PCA] = None
        self.feature_cols = FEATURE_COLS
        self._fitted = False
        self._team_index: dict[str, int] = {}
        self._vectors: Optional[np.ndarray] = None
        self._pca_vectors: Optional[np.ndarray] = None
        self._teams: list[str] = []

    # ------------------------------------------------------------------

    def fit(self, team_df: pd.DataFrame) -> "StyleEncoder":
        """
        Fit scaler (and optionally PCA) on the team feature DataFrame.

        Parameters
        ----------
        team_df : DataFrame with columns including FEATURE_COLS + 'team'
        """
        X = self._extract_matrix(team_df)
        self.scaler.fit(X)
        self._fitted = True

        if self.n_pca_components is not None:
            k = min(self.n_pca_components, X.shape[1], X.shape[0] - 1)
            self.pca = PCA(n_components=k, random_state=42)
            self.pca.fit(self.scaler.transform(X))
            logger.info(
                f"PCA fitted: {k} components explain "
                f"{self.pca.explained_variance_ratio_.sum():.1%} of variance."
            )

        logger.info(f"StyleEncoder fitted on {len(team_df)} teams.")
        return self

    def fit_transform(self, team_df: pd.DataFrame) -> np.ndarray:
        self.fit(team_df)
        return self.transform(team_df)

    def transform(self, team_df: pd.DataFrame) -> np.ndarray:
        """
        Returns a (n_teams, n_features) normalised style matrix.
        Values are in [0, 1]; NaN features are imputed with 0.5 (neutral).
        """
        assert self._fitted, "Call fit() first."
        X = self._extract_matrix(team_df)
        X_scaled = self.scaler.transform(X)
        self._teams = team_df["team"].tolist()
        self._vectors = X_scaled
        self._team_index = {t: i for i, t in enumerate(self._teams)}

        if self.pca is not None:
            self._pca_vectors = self.pca.transform(X_scaled)

        return X_scaled

    def get_vector(self, team_name: str) -> np.ndarray:
        """Return the normalised style vector for a specific team."""
        assert self._vectors is not None, "Call transform() first."
        idx = self._team_index.get(team_name)
        if idx is None:
            raise KeyError(f"Team '{team_name}' not found. Available: {self._teams[:10]}…")
        return self._vectors[idx]

    def get_pca_vector(self, team_name: str) -> np.ndarray:
        """Return the PCA-reduced style vector for a specific team."""
        assert self._pca_vectors is not None, "PCA not fitted or transform not run."
        idx = self._team_index[team_name]
        return self._pca_vectors[idx]

    def to_dataframe(self, use_pca: bool = False) -> pd.DataFrame:
        """Return all style vectors as a labeled DataFrame."""
        assert self._teams, "Call transform() first."
        if use_pca and self._pca_vectors is not None:
            cols = [f"pc{i+1}" for i in range(self._pca_vectors.shape[1])]
            return pd.DataFrame(self._pca_vectors, index=self._teams, columns=cols)
        return pd.DataFrame(self._vectors, index=self._teams, columns=self.feature_cols)

    # ------------------------------------------------------------------
    # Similarity & retrieval
    # ------------------------------------------------------------------

    def most_similar(self, team_name: str, top_k: int = 5) -> pd.DataFrame:
        """Find the k most tactically similar teams (cosine similarity)."""
        assert self._vectors is not None
        vec = self.get_vector(team_name)
        sims = []
        for other, i in self._team_index.items():
            if other == team_name:
                continue
            other_vec = self._vectors[i]
            cos = np.dot(vec, other_vec) / (
                np.linalg.norm(vec) * np.linalg.norm(other_vec) + 1e-8
            )
            sims.append((other, cos))
        sims.sort(key=lambda x: -x[1])
        return pd.DataFrame(sims[:top_k], columns=["team", "similarity"])

    def interpolate(self, team_a: str, team_b: str, alpha: float = 0.5) -> np.ndarray:
        """
        Linear interpolation between two team style vectors.
        alpha=0 → team_a, alpha=1 → team_b.
        Useful for: "Create an agent 70% Man City + 30% Atlético"
        """
        va = self.get_vector(team_a)
        vb = self.get_vector(team_b)
        return (1 - alpha) * va + alpha * vb

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"StyleEncoder saved to {path}.")

    @classmethod
    def load(cls, path: str) -> "StyleEncoder":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"StyleEncoder loaded from {path}.")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_matrix(self, team_df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame, imputing missing values."""
        X = team_df[self.feature_cols].copy().astype(float)

        # Invert features where "lower is better/higher style"
        for col, meta in FEATURE_META.items():
            if meta.get("invert") and col in X.columns:
                # Flip: x → (min + max) - x  so that high value = high style intensity
                X[col] = meta["min"] + meta["max"] - X[col]

        # Impute with column median (neutral style)
        X = X.fillna(X.median())
        return X.values

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"StyleEncoder(n_features={len(self.feature_cols)}, "
            f"n_pca={self.n_pca_components}, {status}, "
            f"n_teams={len(self._teams)})"
        )
