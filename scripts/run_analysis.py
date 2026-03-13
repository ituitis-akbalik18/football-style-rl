"""
Full Analysis Pipeline
======================
Runs all in-depth analyses on extracted style vectors and produces:
  - outputs/ranking_*.png      : top-N teams per feature
  - outputs/similarity_matrix.png : cosine similarity heatmap
  - outputs/radar_iconic.png   : radar charts for iconic teams
  - outputs/feature_importance.png : ANOVA F-score per feature
  - outputs/archetype_boxplots.png : per-feature distribution per archetype
  - outputs/style_distance.png : archetype centroid distance matrix
  - data/processed/analysis_report.csv : full merged analysis table
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import f_oneway

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("=" * 60)
print("Loading processed data...")
team_avg   = pd.read_csv("data/processed/team_avg_features.csv")
archetypes = pd.read_csv("data/processed/team_archetypes.csv")
style_vec  = pd.read_csv("data/processed/style_vectors.csv")

FEATURE_COLS = [
    "ppda", "possession_pct", "pass_directness", "defensive_line_height",
    "dribbles_per90", "counter_attack_speed", "progressive_passes_per90",
    "crosses_per90", "shots_open_play_pct", "high_turnovers_per90",
]
FEAT_LABELS = [
    "PPDA", "Possession%", "Pass Directness", "Def. Line Height",
    "Dribbles/90", "Counter Speed", "Prog. Passes/90",
    "Crosses/90", "Open-Play Shots%", "High Turnovers/90",
]

ARCHETYPE_COLORS = {
    "High-Press Possession": "#1a78cf",
    "Deep-Block Counter":    "#e05c00",
    "Direct Long-Ball":      "#6b4226",
    "Wing-Play Focused":     "#2ca02c",
    "Individual Flair":      "#9467bd",
}

df = team_avg.merge(archetypes[["team", "archetype", "pca_x", "pca_y"]], on="team")
df = df.merge(style_vec[["team"] + FEATURE_COLS].rename(
    columns={f: f + "_norm" for f in FEATURE_COLS}), on="team")

print(f"  Teams: {len(df)}  |  Archetypes: {df['archetype'].nunique()}")


# ─────────────────────────────────────────────
# 2. TOP TEAMS PER FEATURE ranking
# ─────────────────────────────────────────────
print("\n[1/6] Top teams per feature ranking...")

FEAT_CONFIG = {
    "ppda":                    ("PPDA (lower = more pressing)", True),
    "possession_pct":          ("Possession % (higher = more)", False),
    "pass_directness":         ("Pass Directness (higher = more vertical)", False),
    "defensive_line_height":   ("Def. Line Height (higher = deeper block)", False),
    "dribbles_per90":          ("Dribbles per 90", False),
    "progressive_passes_per90": ("Progressive Passes per 90", False),
    "crosses_per90":           ("Crosses per 90", False),
    "counter_attack_speed":    ("Counter-Attack Speed (lower = faster)", True),
}

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for idx, (feat, (title, asc)) in enumerate(FEAT_CONFIG.items()):
    ax = axes[idx]
    top = df[["team", "archetype", feat]].sort_values(feat, ascending=asc).head(12)
    colors = [ARCHETYPE_COLORS.get(a, "#7f7f7f") for a in top["archetype"]]
    bars = ax.barh(range(len(top)), top[feat].values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["team"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.grid(axis="x", alpha=0.3)
    # value labels
    for bar, val in zip(bars, top[feat].values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7)

# Legend
patches = [mpatches.Patch(color=c, label=a) for a, c in ARCHETYPE_COLORS.items()]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Top Teams per Tactical Feature", fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("outputs/ranking_features.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("   -> outputs/ranking_features.png")


# ─────────────────────────────────────────────
# 3. COSINE SIMILARITY MATRIX
# ─────────────────────────────────────────────
print("[2/6] Cosine similarity matrix...")

norm_cols = [f + "_norm" for f in FEATURE_COLS]
X = df[norm_cols].values
sim_matrix = cosine_similarity(X)
sim_df = pd.DataFrame(sim_matrix, index=df["team"].values, columns=df["team"].values)

# Sort by archetype for readability
order = df.sort_values("archetype")["team"].values
sim_sorted = sim_df.loc[order, order]

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.eye(len(order), dtype=bool)
sns.heatmap(
    sim_sorted, ax=ax, cmap="YlOrRd", vmin=0.8, vmax=1.0,
    xticklabels=True, yticklabels=True,
    linewidths=0.3, linecolor="white",
    cbar_kws={"label": "Cosine Similarity", "shrink": 0.7},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
ax.set_title("Team Tactical Style Similarity Matrix\n(sorted by archetype)",
             fontsize=13, fontweight="bold", pad=10)

# Add archetype separators
arch_counts = df.sort_values("archetype")["archetype"].value_counts(sort=False)
arch_order  = df.sort_values("archetype")["archetype"].drop_duplicates().values
cumsum = 0
for arch in arch_order:
    count = (df["archetype"] == arch).sum()
    cumsum += count
    ax.axhline(cumsum, color="black", lw=1.5)
    ax.axvline(cumsum, color="black", lw=1.5)

fig.tight_layout()
fig.savefig("outputs/similarity_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("   -> outputs/similarity_matrix.png")

# Top-5 most similar pairs (excluding same team)
sim_df_copy = sim_df.copy()
np.fill_diagonal(sim_df_copy.values, 0)
pairs = []
for i, t1 in enumerate(df["team"].values):
    for j, t2 in enumerate(df["team"].values):
        if j > i:
            pairs.append((t1, t2, sim_matrix[i, j]))
pairs_df = pd.DataFrame(pairs, columns=["team1", "team2", "similarity"])
pairs_df = pairs_df.sort_values("similarity", ascending=False)
print("   Top 10 most similar team pairs:")
for _, row in pairs_df.head(10).iterrows():
    arch1 = df.loc[df["team"] == row.team1, "archetype"].values[0]
    arch2 = df.loc[df["team"] == row.team2, "archetype"].values[0]
    print(f"   {row.team1:<25} vs {row.team2:<25} | sim={row.similarity:.4f} | {arch1} / {arch2}")

pairs_df.to_csv("data/processed/team_similarity_pairs.csv", index=False)


# ─────────────────────────────────────────────
# 4. ICONIC TEAM RADAR CHARTS
# ─────────────────────────────────────────────
print("[3/6] Iconic team radar charts...")

ICONIC_TEAMS = [
    "Barcelona", "Real Madrid", "Atlético Madrid",
    "Manchester City", "Liverpool",
    "Spain", "Germany", "Brazil",
    "France", "Argentina",
]
# Filter to teams present in our dataset
iconic_present = [t for t in ICONIC_TEAMS if t in df["team"].values]
# Fill up to 9 with top teams from each archetype if needed
if len(iconic_present) < 6:
    extra = df.groupby("archetype").first().reset_index()["team"].tolist()
    for t in extra:
        if t not in iconic_present:
            iconic_present.append(t)
        if len(iconic_present) >= 9:
            break

n = len(iconic_present)
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows),
                          subplot_kw={"polar": True})
axes = np.array(axes).flatten()

N = len(FEATURE_COLS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for i, team in enumerate(iconic_present):
    ax = axes[i]
    row = df[df["team"] == team].iloc[0]
    arch = row["archetype"]
    color = ARCHETYPE_COLORS.get(arch, "#7f7f7f")
    vec = row[[f + "_norm" for f in FEATURE_COLS]].values.tolist()
    vec_closed = vec + [vec[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_ticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEAT_LABELS, size=7)

    # Average reference (grey)
    avg_vec = style_vec[FEATURE_COLS].mean().values.tolist()
    scaler = MinMaxScaler()
    avg_norm = scaler.fit_transform(
        style_vec[FEATURE_COLS].values)
    avg_mean = avg_norm.mean(axis=0).tolist()
    avg_closed = avg_mean + [avg_mean[0]]
    ax.plot(angles, avg_closed, color="grey", lw=1, linestyle="--", alpha=0.4)
    ax.fill(angles, avg_closed, color="grey", alpha=0.08)

    ax.plot(angles, vec_closed, color=color, lw=2)
    ax.fill(angles, vec_closed, color=color, alpha=0.25)

    ax.set_title(f"{team}\n({arch})", size=10, fontweight="bold", pad=12)

for j in range(len(iconic_present), len(axes)):
    axes[j].set_visible(False)

patches = [mpatches.Patch(color=c, label=a) for a, c in ARCHETYPE_COLORS.items()]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Iconic Teams — Tactical Style Radar (grey = global average)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("outputs/radar_iconic.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   -> outputs/radar_iconic.png ({len(iconic_present)} teams: {iconic_present})")


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (ANOVA F-score)
# ─────────────────────────────────────────────
print("[4/6] Feature importance via ANOVA F-score...")

norm_cols = [f + "_norm" for f in FEATURE_COLS]
groups_per_feat = {}
for feat in norm_cols:
    groups_per_feat[feat] = [
        df.loc[df["archetype"] == a, feat].dropna().values
        for a in df["archetype"].unique()
    ]

f_scores = {}
p_values = {}
for feat, groups in groups_per_feat.items():
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        f_scores[feat] = 0; p_values[feat] = 1
        continue
    f, p = f_oneway(*groups)
    f_scores[feat] = f
    p_values[feat] = p

importance_df = pd.DataFrame({
    "feature": [f.replace("_norm", "") for f in norm_cols],
    "f_score": [f_scores[f] for f in norm_cols],
    "p_value": [p_values[f] for f in norm_cols],
    "label":   FEAT_LABELS,
}).sort_values("f_score", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ["#d62728" if p < 0.05 else "#aec7e8"
              for p in importance_df["p_value"]]
bars = ax.barh(importance_df["label"], importance_df["f_score"],
               color=colors_bar, edgecolor="white")
ax.set_xlabel("ANOVA F-score (higher = better separates archetypes)", fontsize=10)
ax.set_title("Feature Importance for Archetype Separation\n(red = p < 0.05)",
             fontsize=12, fontweight="bold")
for bar, val, p in zip(bars, importance_df["f_score"], importance_df["p_value"]):
    label = f"{val:.1f}{'*' if p < 0.05 else ''}"
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("   -> outputs/feature_importance.png")
print("   Top 5 discriminating features:")
for _, r in importance_df.sort_values("f_score", ascending=False).head(5).iterrows():
    print(f"   {r.label:<25}: F={r.f_score:.2f}, p={r.p_value:.4f}")


# ─────────────────────────────────────────────
# 6. ARCHETYPE BOXPLOTS
# ─────────────────────────────────────────────
print("[5/6] Archetype distribution boxplots...")

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

arch_order = list(ARCHETYPE_COLORS.keys())

for idx, (feat, label) in enumerate(zip(FEATURE_COLS, FEAT_LABELS)):
    ax = axes[idx]
    data_per_arch = [df.loc[df["archetype"] == a, feat].dropna().values
                     for a in arch_order]
    bp = ax.boxplot(data_per_arch, patch_artist=True, notch=False,
                    medianprops={"color": "black", "lw": 2})
    for patch, arch in zip(bp["boxes"], arch_order):
        patch.set_facecolor(ARCHETYPE_COLORS[arch])
        patch.set_alpha(0.7)
    ax.set_xticklabels([a.replace(" ", "\n") for a in arch_order], fontsize=6)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Feature Distributions Across Tactical Archetypes",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/archetype_boxplots.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("   -> outputs/archetype_boxplots.png")


# ─────────────────────────────────────────────
# 7. ARCHETYPE CENTROID DISTANCE
# ─────────────────────────────────────────────
print("[6/6] Archetype centroid distance matrix...")

centroids = df.groupby("archetype")[[f + "_norm" for f in FEATURE_COLS]].mean()
cent_sim = cosine_similarity(centroids.values)
cent_dist = 1 - cent_sim

dist_df = pd.DataFrame(cent_dist,
    index=centroids.index, columns=centroids.index)

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.eye(len(dist_df), dtype=bool)
sns.heatmap(dist_df, ax=ax, cmap="Blues", annot=True, fmt=".3f",
            linewidths=0.5, linecolor="white", mask=mask,
            cbar_kws={"label": "Cosine Distance (1 - similarity)"})
ax.set_title("Tactical Distance Between Archetypes\n(0=identical, higher=more different)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/archetype_distance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("   -> outputs/archetype_distance.png")
print("\n   Archetype Centroid Distances:")
print(dist_df.round(4).to_string())


# ─────────────────────────────────────────────
# 8. FULL ANALYSIS REPORT
# ─────────────────────────────────────────────
report = df[["team", "archetype"] + FEATURE_COLS].copy()
# Add ranks per feature
for feat in FEATURE_COLS:
    report[f"rank_{feat}"] = report[feat].rank(ascending=True, na_option="bottom").astype(int)
report.to_csv("data/processed/analysis_report.csv", index=False)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nOutputs generated:")
for f in sorted(os.listdir("outputs")):
    print(f"  outputs/{f}")
print(f"\nData saved:")
print(f"  data/processed/analysis_report.csv")
print(f"  data/processed/team_similarity_pairs.csv")

# Final summary table
print("\n" + "=" * 60)
print("ARCHETYPE SUMMARY")
print("=" * 60)
for arch in arch_order:
    sub = df[df["archetype"] == arch]
    teams = ", ".join(sorted(sub["team"].values))
    print(f"\n{arch} ({len(sub)} teams)")
    print(f"  Teams: {teams}")
    means = sub[FEATURE_COLS].mean()
    print(f"  PPDA={means.ppda:.2f} | Poss={means.possession_pct:.1f}% | "
          f"Directness={means.pass_directness:.3f} | "
          f"Dribbles/90={means.dribbles_per90:.1f}")
