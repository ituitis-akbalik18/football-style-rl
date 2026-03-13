"""
StatsBomb Open Data Loader
--------------------------
Wraps statsbombpy to load competitions, matches, and events.
All data is free via the StatsBomb Open Data GitHub repository.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_available_competitions() -> pd.DataFrame:
    """Return all competitions available in StatsBomb Open Data."""
    from statsbombpy import sb
    comps = sb.competitions()
    logger.info(f"Found {len(comps)} competition-seasons.")
    return comps


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return all matches for a given competition + season."""
    from statsbombpy import sb
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    logger.info(f"Loaded {len(matches)} matches for comp={competition_id}, season={season_id}.")
    return matches


def get_events(match_id: int) -> pd.DataFrame:
    """
    Return all events for a single match.
    Includes passes, shots, dribbles, pressures, ball receipts, etc.
    """
    from statsbombpy import sb
    events = sb.events(match_id=match_id)
    return events


def load_all_events_for_season(
    competition_id: int,
    season_id: int,
    max_matches: Optional[int] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all events and matches for an entire competition-season.

    Returns
    -------
    matches : pd.DataFrame
    events  : pd.DataFrame  (all events concatenated, with match_id column)
    """
    from statsbombpy import sb
    from tqdm import tqdm

    matches = get_matches(competition_id, season_id)
    match_ids = matches["match_id"].tolist()

    if max_matches is not None:
        match_ids = match_ids[:max_matches]

    all_events = []
    iterator = tqdm(match_ids, desc="Loading events") if verbose else match_ids

    for mid in iterator:
        try:
            ev = sb.events(match_id=mid)
            ev["match_id"] = mid
            all_events.append(ev)
        except Exception as e:
            logger.warning(f"Failed to load match {mid}: {e}")

    events = pd.concat(all_events, ignore_index=True)
    logger.info(f"Total events loaded: {len(events):,}")
    return matches, events


# ---------------------------------------------------------------------------
# Convenience: list the most useful open competitions
# ---------------------------------------------------------------------------
RECOMMENDED_COMPETITIONS = {
    # La Liga — multiple seasons to give opponents enough matches
    "La Liga 2015/16": {"competition_id": 11, "season_id": 27},
    "La Liga 2016/17": {"competition_id": 11, "season_id": 2},
    "La Liga 2017/18": {"competition_id": 11, "season_id": 1},
    "La Liga 2018/19": {"competition_id": 11, "season_id": 4},
    "La Liga 2019/20": {"competition_id": 11, "season_id": 42},
    "La Liga 2020/21": {"competition_id": 11, "season_id": 90},
    # Champions League — broad European club coverage
    "Champions League 2018/19": {"competition_id": 16, "season_id": 4},
    "Champions League 2017/18": {"competition_id": 16, "season_id": 1},
    "Champions League 2016/17": {"competition_id": 16, "season_id": 2},
    # International tournaments — national teams
    "World Cup 2022": {"competition_id": 43, "season_id": 106},
    "World Cup 2018": {"competition_id": 43, "season_id": 3},
    "EURO 2020": {"competition_id": 55, "season_id": 43},
    "Copa America 2024": {"competition_id": 223, "season_id": 282},
    # Other leagues
    "Premier League 2003/04": {"competition_id": 2, "season_id": 44},
    "Bundesliga 2023/24": {"competition_id": 9, "season_id": 281},
    "Ligue 1 2015/16": {"competition_id": 7, "season_id": 27},
    "Serie A 2015/16": {"competition_id": 12, "season_id": 27},
}

# Recommended bundle for Phase 1 (good team coverage, manageable size)
PHASE1_BUNDLE = [
    "La Liga 2015/16", "La Liga 2016/17", "La Liga 2017/18",
    "La Liga 2018/19", "La Liga 2019/20", "La Liga 2020/21",
    "Champions League 2018/19",
    "World Cup 2022", "World Cup 2018", "EURO 2020",
]
