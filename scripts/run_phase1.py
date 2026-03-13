"""
Phase 1 — Full Pipeline: Real Data → Style Vectors
====================================================
Run this script to:
  1. Download StatsBomb open data (La Liga, World Cup, etc.)
  2. Extract 10 tactical features per team per match
  3. Aggregate to season-level team profiles
  4. Normalise into style vectors z ∈ [0,1]^10
  5. Cluster into tactical archetypes (K-Means, k=5)
  6. Save outputs to data/processed/ and plots to outputs/

Usage
-----
    python scripts/run_phase1.py
    python scripts/run_phase1.py --competitions "La Liga 2018/19" "World Cup 2022"
    python scripts/run_phase1.py --max-matches 20  # quick test run
"""

import argparse
import logging
import os
import sys
import pickle

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.statsbomb_loader import (
    load_all_events_for_season,
    RECOMMENDED_COMPETITIONS,
)
from src.features.team_features import TeamFeatureExtractor, FEATURE_COLS, FEATURE_META
from src.style_vectors.style_encoder import StyleEncoder
from src.analysis.clustering import ArchetypeClusterer
from src.analysis.visualization import (
    pca_scatter,
    feature_heatmap,
    archetype_bar,
    elbow_silhouette,
    radar_multi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs"
)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Football Style RL – Phase 1 pipeline")
    p.add_argument(
        "--competitions",
        nargs="+",
        default=[
            "La Liga 2018/19", "La Liga 2019/20", "La Liga 2020/21",
            "Champions League 2018/19",
            "World Cup 2022", "EURO 2020",
        ],
        help="Competition names from RECOMMENDED_COMPETITIONS keys",
    )
    p.add_argument(
        "--max-matches", type=int, default=None,
        help="Limit matches per competition (for quick testing)"
    )
    p.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of tactical archetypes"
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download if events already cached"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------

def step1_load_data(competitions: list[str], max_matches, skip_download: bool):
    """Load events for all requested competitions."""
    cache_path = os.path.join(PROCESSED_DIR, "all_events.pkl")
    matches_cache = os.path.join(PROCESSED_DIR, "all_matches.pkl")

    if skip_download and os.path.exists(cache_path):
        logger.info("Loading cached events…")
        events = pd.read_pickle(cache_path)
        matches = pd.read_pickle(matches_cache)
        return matches, events

    all_events_list = []
    all_matches_list = []

    for name in competitions:
        if name not in RECOMMENDED_COMPETITIONS:
            logger.warning(f"Unknown competition '{name}', skipping. "
                           f"Available: {list(RECOMMENDED_COMPETITIONS.keys())}")
            continue

        cfg = RECOMMENDED_COMPETITIONS[name]
        logger.info(f"Loading: {name}  (comp={cfg['competition_id']}, season={cfg['season_id']})")

        try:
            matches, events = load_all_events_for_season(
                competition_id=cfg["competition_id"],
                season_id=cfg["season_id"],
                max_matches=max_matches,
                verbose=True,
            )
            matches["competition_name"] = name
            events["competition_name"] = name
            all_matches_list.append(matches)
            all_events_list.append(events)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    if not all_events_list:
        raise RuntimeError("No data loaded. Check competition names or network connection.")

    all_matches = pd.concat(all_matches_list, ignore_index=True)
    all_events  = pd.concat(all_events_list, ignore_index=True)

    logger.info(f"Total: {len(all_matches)} matches, {len(all_events):,} events loaded.")

    all_events.to_pickle(cache_path)
    all_matches.to_pickle(matches_cache)
    logger.info(f"Cached events to {cache_path}")

    return all_matches, all_events


def step2_extract_features(matches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Extract per-match tactical features for all teams."""
    cache_path = os.path.join(PROCESSED_DIR, "match_features.pkl")

    extractor = TeamFeatureExtractor(events, matches)

    logger.info("Extracting per-match features (this may take a few minutes)…")
    match_features = extractor.compute_all(verbose=True)
    match_features.to_pickle(cache_path)
    match_features.to_csv(cache_path.replace(".pkl", ".csv"), index=False)

    logger.info(f"Saved match features: {len(match_features)} rows → {cache_path}")
    return match_features, extractor


def step3_aggregate(match_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-match features to season team averages."""
    feature_cols = FEATURE_COLS
    team_avg = (
        match_features
        .groupby("team")[feature_cols]
        .mean()
        .reset_index()
    )

    cache_path = os.path.join(PROCESSED_DIR, "team_avg_features.csv")
    team_avg.to_csv(cache_path, index=False)
    logger.info(f"Aggregated {len(team_avg)} teams → {cache_path}")

    # Print summary table
    print("\n" + "="*60)
    print("TEAM FEATURE SUMMARY (top 20)")
    print("="*60)
    display_cols = ["team", "ppda", "possession_pct", "pass_directness",
                    "dribbles_per90", "crosses_per90"]
    available = [c for c in display_cols if c in team_avg.columns]
    print(team_avg[available].round(2).head(20).to_string(index=False))

    return team_avg


def step4_style_vectors(team_avg: pd.DataFrame) -> tuple:
    """Normalise features into style vectors."""
    encoder = StyleEncoder(n_pca_components=3)
    encoder.fit(team_avg)
    z_matrix = encoder.transform(team_avg)
    style_df = encoder.to_dataframe()
    style_df = style_df.reset_index().rename(columns={"index": "team"})

    encoder.save(os.path.join(PROCESSED_DIR, "style_encoder.pkl"))
    style_df.to_csv(os.path.join(PROCESSED_DIR, "style_vectors.csv"), index=False)

    logger.info(f"Style vectors: shape {z_matrix.shape}")

    # Print a few example vectors
    print("\n" + "="*60)
    print("SAMPLE STYLE VECTORS (first 5 teams)")
    print("="*60)
    print(style_df.head(5).round(3).to_string(index=False))

    return encoder, style_df


def step5_cluster(style_df: pd.DataFrame, n_clusters: int):
    """Discover tactical archetypes via K-Means."""
    clusterer = ArchetypeClusterer(n_clusters=n_clusters)

    # Find optimal k first
    logger.info("Running optimal-k analysis…")
    k_df = clusterer.find_optimal_k(style_df, k_range=range(2, 9))
    k_df.to_csv(os.path.join(PROCESSED_DIR, "optimal_k.csv"), index=False)

    # Fit with requested k
    clusterer.fit(style_df)

    labels_df = clusterer.labels_df
    summary   = clusterer.archetype_summary()

    labels_df.to_csv(os.path.join(PROCESSED_DIR, "team_archetypes.csv"), index=False)
    summary.to_csv(os.path.join(PROCESSED_DIR, "archetype_summary.csv"))

    print("\n" + "="*60)
    print("TACTICAL ARCHETYPE ASSIGNMENTS")
    print("="*60)
    for arch, grp in labels_df.groupby("archetype"):
        teams_str = ", ".join(sorted(grp["team"].tolist()))
        print(f"\n  [{arch}] ({len(grp)} teams)")
        print(f"    {teams_str}")

    print("\n" + "="*60)
    print("ARCHETYPE FEATURE MEANS")
    print("="*60)
    print(summary.round(3).to_string())

    return clusterer, labels_df, summary, k_df


def step6_visualise(
    style_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    summary: pd.DataFrame,
    k_df: pd.DataFrame,
    n_clusters: int,
):
    """Generate and save all plots."""
    logger.info("Generating visualisations…")

    # Merge archetype into style_df
    merged = style_df.merge(
        labels_df[["team", "archetype", "pca_x", "pca_y"]], on="team", how="left"
    )

    # 1. PCA scatter
    pca_scatter(
        labels_df,
        save_path=os.path.join(OUTPUT_DIR, "pca_scatter.png"),
    )

    # 2. Feature heatmap
    feature_heatmap(
        merged,
        feature_cols=FEATURE_COLS,
        save_path=os.path.join(OUTPUT_DIR, "feature_heatmap.png"),
        top_n=40,
    )

    # 3. Archetype bar chart
    archetype_bar(
        summary,
        save_path=os.path.join(OUTPUT_DIR, "archetype_bars.png"),
    )

    # 4. Elbow / silhouette
    elbow_silhouette(
        k_df,
        save_path=os.path.join(OUTPUT_DIR, "optimal_k.png"),
    )

    # 5. Radar charts for representative teams (one per archetype)
    representative_teams = (
        labels_df.groupby("archetype")
        .apply(lambda g: g.iloc[0]["team"])
        .tolist()
    )
    available = [t for t in representative_teams if t in merged["team"].values]
    if available:
        radar_multi(
            merged,
            feature_labels=FEATURE_COLS,
            teams=available[:n_clusters],
            save_path=os.path.join(OUTPUT_DIR, "radar_archetypes.png"),
        )

    logger.info(f"All plots saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info("=== Football Style RL — Phase 1: Real Data → Style Vectors ===")
    logger.info(f"Competitions: {args.competitions}")
    logger.info(f"Max matches per competition: {args.max_matches or 'all'}")

    # Step 1: Load data
    matches, events = step1_load_data(
        args.competitions, args.max_matches, args.skip_download
    )

    # Step 2: Extract features
    match_features, extractor = step2_extract_features(matches, events)

    # Step 3: Aggregate to team level
    team_avg = step3_aggregate(match_features)

    # Step 4: Style vectors
    encoder, style_df = step4_style_vectors(team_avg)

    # Step 5: Cluster archetypes
    clusterer, labels_df, summary, k_df = step5_cluster(style_df, args.n_clusters)

    # Step 6: Visualise
    step6_visualise(style_df, labels_df, summary, k_df, args.n_clusters)

    logger.info("\n=== Phase 1 Complete ===")
    logger.info(f"Outputs in: {PROCESSED_DIR}")
    logger.info(f"Plots in:   {OUTPUT_DIR}")
    logger.info(
        "\nNext → Phase 2: Convert style vectors to RL reward/conditioning format."
    )


if __name__ == "__main__":
    main()
