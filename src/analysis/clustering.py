"""
Tactical Archetype Clustering
-------------------------------
Uses K-Means + PCA on the normalised style vectors to discover
and label tactical archetypes (e.g. High-Press Possession, Deep-Block Counter).

Usage
-----
    from src.analysis.clustering import ArchetypeClusterer

    clusterer = ArchetypeClusterer(n_clusters=5)
    clusterer.fit(style_vectors_df)          # DataFrame: teams × features

    print(clusterer.labels_df)               # team → archetype label
    print(clusterer.archetype_summary())     # mean feature per archetype
    clusterer.plot_pca()
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default archetype names mapped to cluster centroids (assigned post-hoc by inspection)
ARCHETYPE_NAMES = [
    "High-Press Possession",
    "Deep-Block Counter",
    "Direct Long-Ball",
    "Wing-Play Focused",
    "Individual Flair",
]


class ArchetypeClusterer:
    """
    Discovers tactical archetypes via K-Means on style vectors.

    Parameters
    ----------
    n_clusters   : number of archetypes to discover (4-6 recommended)
    random_state : reproducibility seed
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.pca2d: Optional[PCA] = None
        self._X: Optional[np.ndarray] = None
        self._X_pca: Optional[np.ndarray] = None
        self.teams_: list[str] = []
        self.feature_cols_: list[str] = []
        self.labels_df: Optional[pd.DataFrame] = None
        self.archetype_map: dict[int, str] = {}

    # ------------------------------------------------------------------

    def fit(self, style_df: pd.DataFrame) -> "ArchetypeClusterer":
        """
        Fit K-Means on the style vector DataFrame.

        Parameters
        ----------
        style_df : DataFrame with 'team' column + feature columns (already [0,1] normalised)
        """
        self.teams_ = style_df["team"].tolist()
        feat_cols = [c for c in style_df.columns if c != "team"]
        self.feature_cols_ = feat_cols

        X = style_df[feat_cols].values.astype(float)
        # Impute any remaining NaN
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        self._X = X

        # K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.random_state,
        )
        cluster_labels = self.kmeans.fit_predict(X)

        # PCA for 2D visualisation
        self.pca2d = PCA(n_components=2, random_state=self.random_state)
        self._X_pca = self.pca2d.fit_transform(X)

        # Auto-assign archetype names based on centroid feature profile
        self.archetype_map = self._assign_archetype_names()

        self.labels_df = pd.DataFrame({
            "team": self.teams_,
            "cluster_id": cluster_labels,
            "archetype": [self.archetype_map[c] for c in cluster_labels],
            "pca_x": self._X_pca[:, 0],
            "pca_y": self._X_pca[:, 1],
        })

        sil = silhouette_score(X, cluster_labels)
        logger.info(
            f"K-Means fitted: {self.n_clusters} clusters, "
            f"silhouette score = {sil:.3f}"
        )
        return self

    def find_optimal_k(
        self, style_df: pd.DataFrame, k_range: range = range(2, 9)
    ) -> pd.DataFrame:
        """
        Evaluate silhouette scores for different k values.
        Useful for choosing the optimal number of archetypes.
        """
        feat_cols = [c for c in style_df.columns if c != "team"]
        X = style_df[feat_cols].fillna(0.5).values.astype(float)

        results = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=20, random_state=self.random_state)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            inertia = km.inertia_
            results.append({"k": k, "silhouette": round(sil, 4), "inertia": round(inertia, 2)})

        df = pd.DataFrame(results)
        logger.info(f"Optimal k analysis:\n{df.to_string(index=False)}")
        return df

    def archetype_summary(self) -> pd.DataFrame:
        """Return mean feature values for each archetype cluster."""
        assert self._X is not None and self.labels_df is not None
        df = self.labels_df[["team", "archetype"]].copy()
        feature_df = pd.DataFrame(self._X, columns=self.feature_cols_)
        combined = pd.concat([df, feature_df], axis=1)
        summary = combined.groupby("archetype")[self.feature_cols_].mean().round(3)
        return summary

    def team_archetype(self, team_name: str) -> str:
        """Return the archetype label for a specific team."""
        assert self.labels_df is not None
        row = self.labels_df[self.labels_df["team"] == team_name]
        if row.empty:
            raise KeyError(f"Team '{team_name}' not found.")
        return row.iloc[0]["archetype"]

    # ------------------------------------------------------------------
    # Automatic archetype naming
    # ------------------------------------------------------------------

    def _assign_archetype_names(self) -> dict[int, str]:
        """
        Heuristically assign human-readable names to clusters
        based on the defining characteristics of each centroid.
        """
        centroids = self.kmeans.cluster_centers_
        feature_cols = self.feature_cols_

        def feat_idx(name):
            try:
                return feature_cols.index(name)
            except ValueError:
                return None

        assignments: dict[int, str] = {}
        used_names: set[str] = set()

        # Feature indices
        i_ppda       = feat_idx("ppda")          # after inversion: high = more pressing
        i_poss       = feat_idx("possession_pct")
        i_directness = feat_idx("pass_directness")
        i_dribbles   = feat_idx("dribbles_per90")
        i_counter    = feat_idx("counter_attack_speed")  # after inversion: high = faster counter
        i_crosses    = feat_idx("crosses_per90")

        # Score each cluster against each archetype profile
        archetype_scores: dict[int, dict[str, float]] = {i: {} for i in range(self.n_clusters)}

        for i, c in enumerate(centroids):
            scores = {}

            # High-Press Possession: high pressing + high possession
            press_score = c[i_ppda] if i_ppda is not None else 0
            poss_score  = c[i_poss] if i_poss is not None else 0
            scores["High-Press Possession"] = (press_score + poss_score) / 2

            # Deep-Block Counter: low possession + fast counter
            low_poss   = 1 - poss_score
            fast_cnt   = c[i_counter] if i_counter is not None else 0
            scores["Deep-Block Counter"] = (low_poss + fast_cnt) / 2

            # Direct Long-Ball: high directness + low possession
            direct     = c[i_directness] if i_directness is not None else 0
            scores["Direct Long-Ball"] = (direct + low_poss) / 2

            # Wing-Play Focused: high crosses
            cross_score = c[i_crosses] if i_crosses is not None else 0
            scores["Wing-Play Focused"] = cross_score

            # Individual Flair: high dribbles
            drib_score  = c[i_dribbles] if i_dribbles is not None else 0
            scores["Individual Flair"] = drib_score

            archetype_scores[i] = scores

        # Greedy assignment: highest-score archetype wins each cluster
        from itertools import permutations
        cluster_ids = list(range(self.n_clusters))
        archetype_pool = list(ARCHETYPE_NAMES[:self.n_clusters])

        # If more clusters than names, use generic names for extras
        while len(archetype_pool) < self.n_clusters:
            archetype_pool.append(f"Style-{len(archetype_pool)+1}")

        best_assignment = {}
        best_score = -np.inf

        for perm in permutations(range(len(archetype_pool)), self.n_clusters):
            total = sum(
                archetype_scores[cid].get(archetype_pool[perm[j]], 0)
                for j, cid in enumerate(cluster_ids)
            )
            if total > best_score:
                best_score = total
                best_assignment = {cid: archetype_pool[perm[j]] for j, cid in enumerate(cluster_ids)}

        logger.info(f"Archetype assignments: {best_assignment}")
        return best_assignment

    def __repr__(self):
        status = "fitted" if self.kmeans else "not fitted"
        return f"ArchetypeClusterer(n_clusters={self.n_clusters}, {status})"
