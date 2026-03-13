"""
Visualisation Suite
--------------------
Produces publication-quality plots for the style analysis:

  - radar_chart()         : team tactical fingerprint radar
  - pca_scatter()         : 2D PCA scatter coloured by archetype
  - feature_heatmap()     : teams × features heatmap
  - archetype_bar()       : mean feature bars per archetype
  - similarity_matrix()   : cosine similarity heatmap between teams
  - elbow_silhouette()    : optimal-k analysis plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, List
import logging
import os

logger = logging.getLogger(__name__)

# Consistent colour palette per archetype
ARCHETYPE_COLORS = {
    "High-Press Possession": "#1a78cf",   # blue
    "Deep-Block Counter":    "#e05c00",   # orange
    "Direct Long-Ball":      "#6b4226",   # brown
    "Wing-Play Focused":     "#2ca02c",   # green
    "Individual Flair":      "#9467bd",   # purple
}
DEFAULT_COLOR = "#7f7f7f"

os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------------------------
# Radar / Spider Chart
# ---------------------------------------------------------------------------

def radar_chart(
    style_vector: np.ndarray,
    feature_labels: List[str],
    team_name: str = "",
    color: str = "#1a78cf",
    ax=None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw a spider/radar chart for a single team's style vector.

    Parameters
    ----------
    style_vector  : 1-D array of normalised [0,1] feature values
    feature_labels: list of feature names (same order as vector)
    """
    N = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]                # close the loop

    values = style_vector.tolist() + [style_vector[0]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    else:
        fig = ax.get_figure()

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid
    ax.set_ylim(0, 1)
    ax.yaxis.set_ticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, size=9)

    # Plot
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)

    # Reference circle
    ax.plot(
        angles,
        [0.5] * (N + 1),
        color="grey",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_title(team_name, size=13, fontweight="bold", pad=15)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Radar chart saved to {save_path}")

    return fig


def radar_multi(
    style_df: pd.DataFrame,
    feature_labels: List[str],
    teams: List[str],
    archetype_col: Optional[str] = "archetype",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Draw radar charts side-by-side for multiple teams."""
    n = len(teams)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                              subplot_kw={"polar": True})
    axes = np.array(axes).flatten() if n > 1 else [axes]

    feat_cols = [c for c in style_df.columns if c in feature_labels or c in feature_labels]

    for i, team in enumerate(teams):
        row = style_df[style_df["team"] == team]
        if row.empty:
            axes[i].set_visible(False)
            continue
        vec = row[feat_cols].values[0]
        arch = row[archetype_col].values[0] if archetype_col and archetype_col in row else None
        color = ARCHETYPE_COLORS.get(arch, DEFAULT_COLOR)
        radar_chart(vec, feature_labels, team_name=team, color=color, ax=axes[i])

    for j in range(len(teams), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Tactical Style Profiles", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# PCA Scatter
# ---------------------------------------------------------------------------

def pca_scatter(
    labels_df: pd.DataFrame,
    archetype_col: str = "archetype",
    highlight_teams: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2D PCA scatter plot of teams coloured by archetype.

    labels_df must have columns: team, pca_x, pca_y, archetype
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    archetypes = labels_df[archetype_col].unique()
    for arch in archetypes:
        sub = labels_df[labels_df[archetype_col] == arch]
        color = ARCHETYPE_COLORS.get(arch, DEFAULT_COLOR)
        ax.scatter(sub["pca_x"], sub["pca_y"], c=color, label=arch,
                   s=80, alpha=0.75, edgecolors="white", linewidths=0.5)
        # Annotate team names (small)
        for _, row in sub.iterrows():
            ax.text(
                row["pca_x"] + 0.01, row["pca_y"] + 0.01,
                row["team"], fontsize=6, alpha=0.7,
            )

    # Highlight specific teams
    if highlight_teams:
        hl = labels_df[labels_df["team"].isin(highlight_teams)]
        ax.scatter(hl["pca_x"], hl["pca_y"], c="gold", s=180, zorder=5,
                   edgecolors="black", linewidths=1.5, marker="*")
        for _, row in hl.iterrows():
            ax.annotate(
                row["team"],
                (row["pca_x"], row["pca_y"]),
                fontsize=9, fontweight="bold",
                xytext=(5, 5), textcoords="offset points",
            )

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("Tactical Style Space (PCA 2D)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PCA scatter saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Feature Heatmap
# ---------------------------------------------------------------------------

def feature_heatmap(
    style_df: pd.DataFrame,
    feature_cols: List[str],
    sort_by_archetype: bool = True,
    archetype_col: str = "archetype",
    top_n: int = 40,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of style feature values across teams.
    Rows = teams, columns = features.
    """
    df = style_df.copy()

    if sort_by_archetype and archetype_col in df.columns:
        df = df.sort_values(archetype_col)

    if len(df) > top_n:
        logger.info(f"Showing top {top_n} teams (sorted).")
        df = df.head(top_n)

    matrix = df[feature_cols].values.astype(float)
    row_labels = df["team"].tolist()

    # Short labels for readability
    col_labels = [c.replace("_per90", "/90").replace("_pct", "%").replace("_", " ").title()
                  for c in feature_cols]

    fig, ax = plt.subplots(figsize=(14, max(6, len(row_labels) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    plt.colorbar(im, ax=ax, label="Normalised Style Intensity [0→1]")
    ax.set_title("Team Tactical Style Heatmap", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Heatmap saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Archetype Summary Bar Chart
# ---------------------------------------------------------------------------

def archetype_bar(
    archetype_summary: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart: mean feature value per archetype.
    archetype_summary = DataFrame from ArchetypeClusterer.archetype_summary()
    """
    archetypes = archetype_summary.index.tolist()
    features = archetype_summary.columns.tolist()
    n_feat = len(features)
    n_arch = len(archetypes)

    x = np.arange(n_feat)
    width = 0.8 / n_arch

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, arch in enumerate(archetypes):
        color = ARCHETYPE_COLORS.get(arch, DEFAULT_COLOR)
        vals = archetype_summary.loc[arch].values
        ax.bar(x + i * width - (n_arch - 1) * width / 2, vals,
               width=width, label=arch, color=color, alpha=0.85)

    col_labels = [c.replace("_per90", "/90").replace("_pct", "%").replace("_", " ").title()
                  for c in features]
    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Normalised Value [0→1]", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Tactical Archetype Feature Profiles", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Elbow / Silhouette Plot
# ---------------------------------------------------------------------------

def elbow_silhouette(
    optimal_k_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Dual-axis plot: inertia (elbow) + silhouette score vs. k.
    optimal_k_df from ArchetypeClusterer.find_optimal_k()
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#1a78cf"
    color2 = "#e05c00"

    ax1.plot(optimal_k_df["k"], optimal_k_df["inertia"], "o-", color=color1, label="Inertia")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia", fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(optimal_k_df["k"], optimal_k_df["silhouette"], "s--", color=color2, label="Silhouette")
    ax2.set_ylabel("Silhouette Score", fontsize=11, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    best_k = optimal_k_df.loc[optimal_k_df["silhouette"].idxmax(), "k"]
    ax2.axvline(x=best_k, color="grey", linestyle=":", linewidth=1.5,
                label=f"Best k={best_k}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    fig.suptitle("Optimal Number of Tactical Archetypes", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
