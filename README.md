# Football Style RL

**Style-Conditioned Multi-Agent Football AI** — bridging real-world match data with diverse tactical opponent agents in Google Research Football.

---

## Project Structure

```
football-style-rl/
├── src/
│   ├── data/
│   │   └── statsbomb_loader.py     # StatsBomb open data loader
│   ├── features/
│   │   └── team_features.py        # 10 tactical feature extractors
│   ├── analysis/
│   │   ├── clustering.py           # K-Means archetype discovery
│   │   └── visualization.py        # Radar, PCA, heatmap plots
│   └── style_vectors/
│       └── style_encoder.py        # Normalisation + PCA embedding
├── scripts/
│   └── run_phase1.py               # Full Phase 1 pipeline
├── data/
│   ├── raw/                        # Downloaded StatsBomb data (gitignored)
│   └── processed/                  # Computed features, style vectors, archetypes
├── outputs/                        # Generated plots
└── requirements.txt
```

---

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Real data → style vectors → archetypes | ✅ |
| **2** | Style vectors → RL reward/conditioning format | 🔜 |
| **3** | RL training with style conditioning (GRF) | 🔜 |

---

## Phase 1: Data → Style Vectors

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# Full pipeline (La Liga 2018/19, World Cup 2022, EURO 2020)
python scripts/run_phase1.py

# Quick test with limited matches
python scripts/run_phase1.py --max-matches 10

# Custom competitions
python scripts/run_phase1.py --competitions "La Liga 2018/19" "Champions League 2021/22"
```

### Extracted Features

| Feature | Description | Style Dimension |
|---------|-------------|-----------------|
| `ppda` | Passes per defensive action | Pressing intensity |
| `possession_pct` | Ball possession % | Control orientation |
| `pass_directness` | Progressive / total passes | Vertical vs lateral |
| `defensive_line_height` | Avg x of defensive actions | High vs deep block |
| `dribbles_per90` | Dribble attempts per 90 min | Individual flair |
| `counter_attack_speed` | Seconds: recovery → shot | Transition speed |
| `progressive_passes_per90` | Passes advancing ≥10 yd | Build-up speed |
| `crosses_per90` | Cross attempts per 90 min | Wing play tendency |
| `shots_open_play_pct` | % shots from open play | Attack pattern |
| `high_turnovers_per90` | Ball losses in opp. third | Risk appetite |

### Tactical Archetypes

| Archetype | Examples |
|-----------|----------|
| High-Press Possession | Man City, Barcelona, Bayern |
| Deep-Block Counter | Atlético Madrid, Inter |
| Direct Long-Ball | Burnley, Stoke City |
| Wing-Play Focused | Liverpool (Klopp) |
| Individual Flair | Brazil NT, Santos |

### Outputs

- `data/processed/match_features.csv` — per-match feature table
- `data/processed/team_avg_features.csv` — season-level team averages
- `data/processed/style_vectors.csv` — normalised [0,1] style vectors
- `data/processed/team_archetypes.csv` — team → archetype mapping
- `data/processed/style_encoder.pkl` — fitted encoder for RL use
- `outputs/pca_scatter.png` — team tactical space visualisation
- `outputs/feature_heatmap.png` — style feature heatmap
- `outputs/radar_archetypes.png` — per-archetype radar charts
- `outputs/archetype_bars.png` — feature profiles per archetype

---

## References

- Kurach et al., *Google Research Football*, AAAI 2020
- Song et al., *GRF_MARL Benchmark*, MIR 2024
- Sun et al., *LCDSP*, arXiv 2025
- Rahimian & Toka, *IRL in Football*, ECML-PKDD 2021
- StatsBomb Open Data: https://github.com/statsbomb/open-data
