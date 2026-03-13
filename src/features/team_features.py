"""
Team-Level Tactical Feature Extraction
---------------------------------------
Computes the 10 core style features from StatsBomb event data:

  1. ppda                  – Passes Per Defensive Action (pressing intensity)
  2. possession_pct        – Average ball possession %
  3. pass_directness       – Ratio of progressive/forward passes
  4. defensive_line_height – Average pitch height of defensive actions
  5. dribbles_per90        – Dribble attempts per 90 minutes
  6. counter_attack_speed  – Avg seconds from recovery to shot/chance
  7. progressive_passes_p90 – Passes that advance ball ≥10 m toward goal
  8. crosses_per90         – Crosses per 90 minutes
  9. shots_open_play_pct   – % shots from open play
 10. high_turnovers_per90  – Ball losses in opponent's third

Usage
-----
    from src.features.team_features import TeamFeatureExtractor

    extractor = TeamFeatureExtractor(events, matches)
    features_df = extractor.compute_all()  # one row per team per match
    team_avg = extractor.aggregate()       # season-level team averages
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# StatsBomb pitch dimensions (yards)
PITCH_LENGTH = 120.0   # x: 0 → 120
PITCH_WIDTH  = 80.0    # y: 0 → 80

# "Opponent's half" starts at x=60 (midfield line)
OPPONENT_HALF_X = 60.0

# For progressive pass: ball must advance toward goal by ≥10 yards,
# and end in opponent's half OR end closer to goal than start.
PROGRESSIVE_DIST_THRESHOLD = 10.0  # yards

# Opponent's defensive third: x ≥ 80 (last 40 yards)
OPPONENT_THIRD_X = 80.0


class TeamFeatureExtractor:
    """
    Computes per-match tactical feature vectors for every team in a season.

    Parameters
    ----------
    events  : pd.DataFrame   – concatenated StatsBomb events (all matches)
    matches : pd.DataFrame   – match metadata (home/away team, possession %)
    """

    def __init__(self, events: pd.DataFrame, matches: pd.DataFrame):
        self.events = events.copy()
        self.matches = matches.copy()
        self._preprocess()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self):
        """Expand nested StatsBomb columns and add helper columns."""
        ev = self.events

        # ---- location (ball position) --------------------------------
        if "location" in ev.columns:
            ev[["loc_x", "loc_y"]] = pd.DataFrame(
                ev["location"].apply(
                    lambda v: v if isinstance(v, (list, tuple)) else [np.nan, np.nan]
                ).tolist(),
                index=ev.index,
            )
        else:
            ev["loc_x"] = np.nan
            ev["loc_y"] = np.nan

        # ---- pass end location ---------------------------------------
        if "pass_end_location" in ev.columns:
            ev[["pass_end_x", "pass_end_y"]] = pd.DataFrame(
                ev["pass_end_location"].apply(
                    lambda v: v if isinstance(v, (list, tuple)) else [np.nan, np.nan]
                ).tolist(),
                index=ev.index,
            )
        else:
            ev["pass_end_x"] = np.nan
            ev["pass_end_y"] = np.nan

        # ---- timestamp → seconds ------------------------------------
        if "timestamp" in ev.columns:
            ev["seconds"] = ev["timestamp"].apply(self._ts_to_seconds)
        else:
            ev["seconds"] = np.nan

        # ---- possession team normalisation ---------------------------
        # StatsBomb stores team names; we need a consistent identifier
        if "possession_team" in ev.columns and isinstance(
            ev["possession_team"].iloc[0], dict
        ):
            ev["possession_team_name"] = ev["possession_team"].apply(
                lambda x: x.get("name", "") if isinstance(x, dict) else x
            )
        elif "possession_team" in ev.columns:
            ev["possession_team_name"] = ev["possession_team"]
        else:
            ev["possession_team_name"] = ""

        if "team" in ev.columns and isinstance(ev["team"].iloc[0], dict):
            ev["team_name"] = ev["team"].apply(
                lambda x: x.get("name", "") if isinstance(x, dict) else x
            )
        elif "team" in ev.columns:
            ev["team_name"] = ev["team"]
        else:
            ev["team_name"] = ""

        self.events = ev

    @staticmethod
    def _ts_to_seconds(ts) -> float:
        """Convert 'HH:MM:SS.ms' timestamp to total seconds."""
        try:
            parts = str(ts).split(":")
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # Individual feature methods (per match, per team)
    # ------------------------------------------------------------------

    def _ppda(self, match_events: pd.DataFrame, team: str) -> float:
        """
        Passes Per Defensive Action in the opponent's half.
        PPDA = opponent passes in own half / team defensive actions in opponent's half

        Lower PPDA → higher pressing intensity.
        """
        # Opponent passes in their own half (x < 60 when opponent has possession)
        opp_passes = match_events[
            (match_events["team_name"] != team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Pass")
            & (match_events["loc_x"] < OPPONENT_HALF_X)
            & (match_events["pass_outcome"].apply(
                lambda x: (x.get("name", "") if isinstance(x, dict) else str(x or "")) not in
                ["Incomplete", "Out", "Pass Offside", "Unknown"]
            ) if "pass_outcome" in match_events.columns else True)
        ]

        # Team defensive actions in opponent's half
        def_action_types = {"Pressure", "Tackle", "Interception", "Block", "Ball Recovery"}
        def_actions = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: (t.get("name", t) if isinstance(t, dict) else t)
                in def_action_types
            ))
            & (match_events["loc_x"] >= OPPONENT_HALF_X)
        ]

        n_def = len(def_actions)
        if n_def == 0:
            return np.nan
        return len(opp_passes) / n_def

    def _possession_pct(self, match_id: int, team: str) -> float:
        """
        Ball possession percentage from match metadata if available,
        otherwise estimated from event counts.
        """
        # Try match-level column first
        match_row = self.matches[self.matches["match_id"] == match_id]
        if not match_row.empty:
            if "home_team" in match_row.columns:
                home_team = match_row.iloc[0]["home_team"]
                if isinstance(home_team, dict):
                    home_team = home_team.get("home_team_name", "")
                away_team = match_row.iloc[0].get("away_team", "")
                if isinstance(away_team, dict):
                    away_team = away_team.get("away_team_name", "")

        # Estimate from possession event counts (StatsBomb tracks possession_team)
        match_ev = self.events[self.events["match_id"] == match_id]
        total = len(match_ev[match_ev["possession_team_name"] != ""])
        if total == 0:
            return np.nan
        team_poss = len(match_ev[match_ev["possession_team_name"] == team])
        return 100.0 * team_poss / total

    def _pass_directness(self, match_events: pd.DataFrame, team: str) -> float:
        """
        Pass Directness Index = progressive passes / total completed passes.
        A progressive pass advances the ball ≥10 yards toward the opponent goal.
        """
        passes = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Pass")
        ].copy()

        if len(passes) == 0:
            return np.nan

        # Filter completed passes
        if "pass_outcome" in passes.columns:
            completed = passes[
                passes["pass_outcome"].apply(
                    lambda x: (x.get("name", "") if isinstance(x, dict) else str(x or ""))
                    not in ["Incomplete", "Out", "Pass Offside", "Unknown"]
                )
            ]
        else:
            completed = passes

        if len(completed) == 0:
            return np.nan

        # Progressive: end_x > start_x by ≥ threshold AND end moves closer to goal
        prog = completed[
            (completed["pass_end_x"] - completed["loc_x"] >= PROGRESSIVE_DIST_THRESHOLD)
            & (completed["pass_end_x"] >= OPPONENT_HALF_X)
        ]

        return len(prog) / len(completed)

    def _defensive_line_height(self, match_events: pd.DataFrame, team: str) -> float:
        """
        Average x-coordinate of the team's defensive actions (own half).
        Higher value = higher defensive line.
        Normalised to [0, 1] over pitch length.
        """
        def_action_types = {"Tackle", "Interception", "Block", "Clearance", "Ball Recovery"}
        def_acts = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: (t.get("name", t) if isinstance(t, dict) else t)
                in def_action_types
            ))
            & (match_events["loc_x"].notna())
            & (match_events["loc_x"] < OPPONENT_HALF_X)  # own half
        ]

        if len(def_acts) == 0:
            return np.nan
        return def_acts["loc_x"].mean() / PITCH_LENGTH

    def _dribbles_per90(self, match_events: pd.DataFrame, team: str, minutes: float) -> float:
        """Dribble attempts per 90 minutes."""
        if minutes <= 0:
            return np.nan
        dribbles = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Dribble")
        ]
        return len(dribbles) / minutes * 90.0

    def _counter_attack_speed(self, match_events: pd.DataFrame, team: str) -> float:
        """
        Average seconds from ball recovery (in own half) to next shot/chance.
        Proxy for transition/counter-attack speed.
        Lower value = faster counter-attacks.
        """
        recoveries = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Ball Recovery")
            & (match_events["loc_x"] < OPPONENT_HALF_X)
        ].copy()

        if len(recoveries) == 0:
            return np.nan

        shots = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Shot")
        ].copy()

        if len(shots) == 0:
            return np.nan

        speeds = []
        for _, rec in recoveries.iterrows():
            rec_idx = rec.name
            rec_time = rec["seconds"]
            if pd.isna(rec_time):
                continue
            # Find next shot by this team after the recovery
            next_shots = shots[
                (shots.index > rec_idx) & (shots["seconds"] > rec_time)
            ]
            if len(next_shots) == 0:
                continue
            next_shot_time = next_shots.iloc[0]["seconds"]
            elapsed = next_shot_time - rec_time
            # Only count "quick" transitions (< 15 seconds, realistic for counter)
            if 0 < elapsed < 15:
                speeds.append(elapsed)

        return np.mean(speeds) if speeds else np.nan

    def _progressive_passes_per90(
        self, match_events: pd.DataFrame, team: str, minutes: float
    ) -> float:
        """Progressive passes per 90 minutes."""
        if minutes <= 0:
            return np.nan
        passes = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Pass")
            & (match_events["pass_end_x"] - match_events["loc_x"] >= PROGRESSIVE_DIST_THRESHOLD)
            & (match_events["pass_end_x"] >= OPPONENT_HALF_X)
        ]
        return len(passes) / minutes * 90.0

    def _crosses_per90(
        self, match_events: pd.DataFrame, team: str, minutes: float
    ) -> float:
        """Crosses per 90 minutes (passes flagged as cross in StatsBomb data)."""
        if minutes <= 0:
            return np.nan
        crosses = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Pass")
            & (match_events.get("pass_cross", pd.Series([False] * len(match_events))).fillna(False))
        ]
        return len(crosses) / minutes * 90.0

    def _shots_open_play_pct(self, match_events: pd.DataFrame, team: str) -> float:
        """Percentage of shots that come from open play (not set pieces)."""
        shots = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: t.get("name", t) if isinstance(t, dict) else t
            ) == "Shot")
        ]
        if len(shots) == 0:
            return np.nan

        if "shot_type" in shots.columns:
            open_play = shots[
                shots["shot_type"].apply(
                    lambda x: (x.get("name", "") if isinstance(x, dict) else str(x or ""))
                    == "Open Play"
                )
            ]
        else:
            return np.nan

        return 100.0 * len(open_play) / len(shots)

    def _high_turnovers_per90(
        self, match_events: pd.DataFrame, team: str, minutes: float
    ) -> float:
        """Ball losses in the opponent's third per 90 minutes (high press risk metric)."""
        if minutes <= 0:
            return np.nan

        # Miscontrol or dispossessed in opponent's third
        turnovers = match_events[
            (match_events["team_name"] == team)
            & (match_events["type"].apply(
                lambda t: (t.get("name", t) if isinstance(t, dict) else t)
                in {"Miscontrol", "Dispossessed"}
            ))
            & (match_events["loc_x"] >= OPPONENT_THIRD_X)
        ]
        return len(turnovers) / minutes * 90.0

    # ------------------------------------------------------------------
    # Match duration helper
    # ------------------------------------------------------------------

    @staticmethod
    def _match_minutes(match_events: pd.DataFrame) -> float:
        """Estimate match duration in minutes from the last event timestamp."""
        max_sec = match_events["seconds"].max()
        if pd.isna(max_sec):
            return 90.0  # assume full match
        return max(max_sec / 60.0, 45.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_match_features(self, match_id: int, team: str) -> dict:
        """Compute all features for a single team in a single match."""
        match_ev = self.events[self.events["match_id"] == match_id].copy()
        minutes = self._match_minutes(match_ev)

        return {
            "match_id": match_id,
            "team": team,
            "minutes": round(minutes, 1),
            "ppda": self._ppda(match_ev, team),
            "possession_pct": self._possession_pct(match_id, team),
            "pass_directness": self._pass_directness(match_ev, team),
            "defensive_line_height": self._defensive_line_height(match_ev, team),
            "dribbles_per90": self._dribbles_per90(match_ev, team, minutes),
            "counter_attack_speed": self._counter_attack_speed(match_ev, team),
            "progressive_passes_per90": self._progressive_passes_per90(match_ev, team, minutes),
            "crosses_per90": self._crosses_per90(match_ev, team, minutes),
            "shots_open_play_pct": self._shots_open_play_pct(match_ev, team),
            "high_turnovers_per90": self._high_turnovers_per90(match_ev, team, minutes),
        }

    def compute_all(self, verbose: bool = True) -> pd.DataFrame:
        """
        Compute features for every (match, team) pair in the dataset.
        Returns one row per (match_id, team).
        """
        from tqdm import tqdm

        match_ids = self.events["match_id"].unique()
        rows = []

        iterator = tqdm(match_ids, desc="Extracting features") if verbose else match_ids

        for mid in iterator:
            match_ev = self.events[self.events["match_id"] == mid]
            teams = match_ev["team_name"].dropna().unique()
            for team in teams:
                if team == "":
                    continue
                row = self.compute_match_features(mid, team)
                rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Feature extraction complete: {len(df)} rows ({df['team'].nunique()} teams).")
        return df

    def aggregate(
        self,
        match_features: Optional[pd.DataFrame] = None,
        min_matches: int = 3,
    ) -> pd.DataFrame:
        """
        Aggregate per-match features into season-level team averages.

        Parameters
        ----------
        match_features : pre-computed DataFrame from compute_all(); computed if None
        min_matches    : minimum matches required to include a team

        Returns
        -------
        DataFrame with one row per team, columns = feature names
        """
        if match_features is None:
            match_features = self.compute_all()

        feature_cols = [
            "ppda", "possession_pct", "pass_directness",
            "defensive_line_height", "dribbles_per90",
            "counter_attack_speed", "progressive_passes_per90",
            "crosses_per90", "shots_open_play_pct", "high_turnovers_per90",
        ]

        # Count actual matches per team (not NaN-filtered feature counts)
        match_counts = match_features.groupby("team")["match_id"].count().rename("n_matches")

        agg = (
            match_features
            .groupby("team")[feature_cols]
            .mean()
            .round(4)
        )
        agg = agg.reset_index()
        agg = agg.merge(match_counts, on="team")

        # Filter teams with enough data
        agg = agg[agg["n_matches"] >= min_matches].copy()

        style_df = agg[["team"] + feature_cols]

        n_before = len(agg) + (match_counts < min_matches).sum()
        logger.info(f"Aggregated {len(style_df)} teams with ≥{min_matches} matches.")
        return style_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature metadata (used downstream for normalisation & visualisation)
# ---------------------------------------------------------------------------
FEATURE_META = {
    "ppda": {
        "label": "PPDA",
        "description": "Passes per defensive action (lower = more pressing)",
        "invert": True,   # lower is "more intense" — invert for radar charts
        "min": 5.0,
        "max": 25.0,
    },
    "possession_pct": {
        "label": "Possession %",
        "description": "Average ball possession percentage",
        "invert": False,
        "min": 30.0,
        "max": 75.0,
    },
    "pass_directness": {
        "label": "Pass Directness",
        "description": "Ratio of progressive to total completed passes",
        "invert": False,
        "min": 0.1,
        "max": 0.5,
    },
    "defensive_line_height": {
        "label": "Def. Line Height",
        "description": "Normalised pitch position of defensive actions",
        "invert": False,
        "min": 0.2,
        "max": 0.5,
    },
    "dribbles_per90": {
        "label": "Dribbles/90",
        "description": "Dribble attempts per 90 minutes",
        "invert": False,
        "min": 5.0,
        "max": 25.0,
    },
    "counter_attack_speed": {
        "label": "Counter Speed (s)",
        "description": "Avg seconds from recovery to shot",
        "invert": True,   # lower = faster counter
        "min": 3.0,
        "max": 12.0,
    },
    "progressive_passes_per90": {
        "label": "Progressive Passes/90",
        "description": "Passes advancing ball ≥10 yards per 90 min",
        "invert": False,
        "min": 20.0,
        "max": 70.0,
    },
    "crosses_per90": {
        "label": "Crosses/90",
        "description": "Cross attempts per 90 minutes",
        "invert": False,
        "min": 5.0,
        "max": 30.0,
    },
    "shots_open_play_pct": {
        "label": "Open Play Shots %",
        "description": "Percentage of shots from open play",
        "invert": False,
        "min": 50.0,
        "max": 90.0,
    },
    "high_turnovers_per90": {
        "label": "High Turnovers/90",
        "description": "Ball losses in opponent's third per 90 min",
        "invert": False,
        "min": 1.0,
        "max": 10.0,
    },
}

FEATURE_COLS = list(FEATURE_META.keys())
