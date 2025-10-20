import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re


tick_fontsize = 14
label_fontsize = 14
legend_fontsize = 14
title_fontsize = 16

with open("/p/crash/amaas/safety_hardening_frame_stack_prioritized_5_cycle_old_reward/ego_model_pool.json") as f:
    ego_pool_data = json.load(f)
with open("/p/crash/amaas/safety_hardening_frame_stack_prioritized_5_cycle_old_reward/npc_model_pool.json") as f:
    npc_pool_data = json.load(f)

def parse_tag_to_int(tag: str) -> float:
    m = re.search(r"\d+", str(tag))
    return float(m.group()) if m else np.nan

def build_pool_mats(pool_json):
    """Return:
        meta_mat: (n_rows, 6) -> [cycle, Sa, Sb, model_idx, opp_model_num, opp_elo]
        elo_mat:  (n_rows, max_models) -> per-row ELOs, NaN if model not present
        prob_mat: (n_rows, max_models) -> per-row probabilities, NaN if not present
        spans:    list of (start_idx, end_idx, cycle_int)
        max_models: int
    """
    cycles = sorted(pool_json.keys(), key=int)

    # Determine max models across all rows
    max_models = 0
    for c in cycles:
        for k in pool_json[c].keys():
            entry = pool_json[c][k]
            max_models = max(max_models, len(entry["model_elo"]), len(entry["model_probability"]))

    rows_meta = []
    rows_elo = []
    rows_prob = []
    spans = []
    row_idx = 0

    for c in cycles:
        eval_keys = sorted(pool_json[c].keys(), key=int)
        start = row_idx
        for ek in eval_keys:
            e = pool_json[c][ek]
            meta = np.array([
                int(c),
                float(e["Sa"]),
                float(e["Sb"]),
                float(e["model_idx"]),
                parse_tag_to_int(e["opponent_model"]),
                float(e["opponent_elo"])
            ], dtype=float)

            elo_row = np.full(max_models, np.nan, dtype=float)
            for i, val in enumerate(e["model_elo"]):
                elo_row[i] = float(val)

            prob_row = np.full(max_models, np.nan, dtype=float)
            for i, val in enumerate(e["model_probability"]):
                prob_row[i] = float(val)

            rows_meta.append(meta)
            rows_elo.append(elo_row)
            rows_prob.append(prob_row)
            row_idx += 1
        spans.append((start, row_idx - 1, int(c)))

    meta_mat = np.vstack(rows_meta) if rows_meta else np.zeros((0, 6))
    elo_mat = np.vstack(rows_elo) if rows_elo else np.zeros((0, max_models))
    prob_mat = np.vstack(rows_prob) if rows_prob else np.zeros((0, max_models))
    return meta_mat, elo_mat, prob_mat, spans, max_models


def shade(ax_, spans):
    for s, e, _ in spans:
        ax_.axvspan(s, e + 1, alpha=0.12, zorder=0)

def plot_prob(ax, prob_mat, n_models, label_prefix):
    # Plot only series that have at least one finite value
    for i in range(n_models):
        y = prob_mat[:, i]
        finite = np.isfinite(y)
        if not finite.any():
            continue
        line, = ax.plot(y, label=f"{label_prefix}{i+1}", zorder=5, linewidth=1.8)
        first_idx = np.argmax(finite)
        ax.plot(first_idx, y[first_idx], marker="o", markersize=5, zorder=6)

def draw_uniform_refs(ax, prob_mat, spans):
    """
    For each cycle span, compute the max number of active models (finite probs)
    and draw a dashed horizontal line at y = 1/m across that span.
    """
    active_counts = np.sum(np.isfinite(prob_mat), axis=1)  # per-row model count
    for s, e, c in spans:
        if e < s: 
            continue
        m = int(np.nanmax(active_counts[s:e+1]))  # models present in this span
        if m < 1:
            continue
        y = 1.0 / m
        ax.hlines(y, s, e + 1, linestyles="dashed", linewidth=1.5, zorder=4)
        # small label so we know which uniform level it is
        ax.text(s + 1, y + 0.02, f"1/{m}", fontsize=10, va="bottom")

def prob_variance_row(prob_row: np.ndarray) -> float:
    """
    Population variance of a single row of probabilities vs uniform mean (1/m),
    where m is the number of active (finite) model probabilities in this row.
    Returns np.nan if no active models.
    """
    mask = np.isfinite(prob_row)
    m = int(mask.sum())
    if m == 0:
        return np.nan
    mu = 1.0 / m
    diffsq = (prob_row[mask] - mu) ** 2
    # population variance around fixed mean = (1/m) * sum(...)
    return float(diffsq.sum() / m)

def prob_variance_series(prob_mat: np.ndarray) -> np.ndarray:
    """
    Row-wise population variance vs uniform mean for an entire matrix of
    shape (n_rows, n_models). Handles NaNs for not-yet-present models.
    """
    variances = np.full(prob_mat.shape[0], np.nan, dtype=float)
    for r in range(prob_mat.shape[0]):
        variances[r] = prob_variance_row(prob_mat[r])
    return variances

def prob_variance_by_cycle(prob_mat: np.ndarray, spans, reducer=np.nanmean):
    """
    Aggregate variance per cycle.
    spans: list of (start_idx, end_idx, cycle_int)
    reducer: e.g., np.nanmean, np.nanmedian, np.nanmax
    Returns dict {cycle_int: aggregated_variance}
    """
    row_vars = prob_variance_series(prob_mat)
    out = {}
    for s, e, c in spans:
        out[c] = reducer(row_vars[s:e+1])
    return out

def shade(ax, spans):
    """Lightly shade each cycle span."""
    for s, e, _ in spans:
        if e >= s:
            ax.axvspan(s, e + 1, alpha=0.12, zorder=0)

def draw_uniform_refs(ax, prob_mat, spans):
    """
    For each cycle, draw a dashed line at 1/m where m = max number of active models
    (active = finite probabilities) within that cycle span.
    """
    active_counts = np.sum(np.isfinite(prob_mat), axis=1)  # models present per row
    for s, e, _ in spans:
        if e < s:
            continue
        m = int(np.nanmax(active_counts[s:e+1]))
        if m >= 1:
            y = 1.0 / m
            ax.hlines(y, s, e + 1, linestyles="dashed", linewidth=1.0, zorder=4, color = 'black')
            ax.text(s + 1, y + 0.1, f"1/{m}", fontsize=10, va="bottom")

def padded_ylim_from_lines(ax, pad=25, pct=(1, 99)):
    """Set y-limits based on data in line artists, ignoring NaNs."""
    vals = []
    for ln in ax.get_lines():
        y = ln.get_ydata(orig=False)
        y = y[np.isfinite(y)]
        if y.size:
            vals.append(y)
    if vals:
        vals = np.concatenate(vals)
        lo, hi = np.percentile(vals, pct)
        ax.set_ylim(lo - pad, hi + pad)

def plot_series(ax, mat, n_models, label_prefix, linewidth=1.8, mark_first=True, z=5):
    """
    Plot each column i of `mat` (shape (rows, n_models)).
    Skips columns that are entirely NaN. Optionally marks the first finite point.
    """
    for i in range(n_models):
        y = mat[:, i]
        finite = np.isfinite(y)
        if not finite.any():
            continue
        ax.plot(y, label=f"{label_prefix}{i+1}", linewidth=linewidth, zorder=z)
        if mark_first:
            first_idx = np.argmax(finite)
            ax.plot(first_idx, y[first_idx], marker="o", markersize=5, zorder=z+1)

def plot_ego_elo_and_probs(axes, ego_elo, ego_prob, ego_spans, ego_models):
    """
    Create a 2-row subplot:
      Top: Ego ELO per model (E0..E{n-1})
      Bottom: Ego model selection probabilities per model, with uniform reference lines
    """

    # --- Top: ELO ---
    shade(axes[0], ego_spans)
    plot_series(axes[0], ego_elo, ego_models, label_prefix="E")
    axes[0].set_title("Ego Pool — ELO per Model")
    axes[0].set_ylabel("ELO")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")
    padded_ylim_from_lines(axes[0])  # automatic y range for ELO

    # --- Bottom: probabilities ---
    shade(axes[1], ego_spans)
    draw_uniform_refs(axes[1], ego_prob, ego_spans)
    plot_series(axes[1], ego_prob, ego_models, label_prefix="E")
    axes[1].set_title("Ego Pool — Model Selection Probabilities")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Evaluation (row index)")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.show()

from matplotlib import patches as mpatches

def cycle_colors(spans, cmap_name="tab20"):
    """
    Returns a list of RGBA colors, one per span, using a colormap.
    Order follows spans (so cycle 1 gets colors[0], cycle 2 -> colors[1], etc.).
    """
    n = len(spans)
    cmap = plt.get_cmap(cmap_name)
    # spread colors nicely across the map
    return [cmap(x) for x in np.linspace(0, 1, max(n, 2))[:n]]

def shade_by_cycle(ax, spans, colors=None, alpha=0.12, z=0, add_legend=False, title_prefix="Cycle"):
    """
    Shades each (start,end) span with its own color.
    If add_legend=True, adds a legend showing band colors per cycle.
    """
    if colors is None:
        colors = cycle_colors(spans)

    handles = []
    for (i, (s, e, c)) in enumerate(spans):
        if e < s:
            continue
        ax.axvspan(s, e + 1, color=colors[i], alpha=alpha, zorder=z)
        if add_legend:
            handles.append(mpatches.Patch(facecolor=colors[i], alpha=alpha, label=f"{title_prefix} {c}"))

    if add_legend and handles:
        # place a compact legend for the bands; adjust loc as you like
        ax.legend(handles=handles, loc="upper left", frameon=False, title="Spans", fontsize=9)

# --- call it ---
# ego_meta, ego_elo, ego_prob, ego_spans, ego_models = build_pool_mats(ego_pool_data)


# ego_meta, ego_elo, ego_prob, ego_spans, ego_models = build_pool_mats(ego_pool_data)
# npc_meta, npc_elo, npc_prob, npc_spans, npc_models = build_pool_mats(npc_pool_data)

# fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

# # Shade first so lines are drawn on top
# shade_by_cycle(ax[0], ego_spans, alpha=0.12)
# shade_by_cycle(ax[1], npc_spans, alpha=0.12)

# draw_uniform_refs(ax[0], ego_prob, ego_spans)
# draw_uniform_refs(ax[1], npc_prob, npc_spans)

# # --- Plot PROBABILITIES instead of ELOs ---
# plot_prob(ax[0], ego_prob, ego_models, "E")
# plot_prob(ax[1], npc_prob, npc_models, "V")

# ax[0].set_title("Ego Pool — Model Selection Probabilities")
# ax[1].set_title("NPC Pool — Model Selection Probabilities")

# for a in ax:
#     a.grid(True)
#     a.legend(loc="upper right")
#     a.set_ylabel("Probability")
#     a.set_ylim(0.0, 1.05)  # probabilities in [0,1]
#     a.set_xticks([0, 10, 30, 60, 100, 150])

# ax[1].set_xlabel("Evaluation (row index)")

# plt.tight_layout()
# plt.show()
ego_meta, ego_elo, ego_prob, ego_spans, ego_models = build_pool_mats(ego_pool_data)
npc_meta, npc_elo, npc_prob, npc_spans, npc_models = build_pool_mats(npc_pool_data)

# --- Variances (row-wise) ---
ego_var = prob_variance_series(ego_prob)
npc_var = prob_variance_series(npc_prob)

# --- 3-row subfigure ---
fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

# Row 1: Ego ELO
shade_by_cycle(ax[0], ego_spans, alpha=0.12)
plot_series(ax[0], ego_elo, ego_models, label_prefix="E")
ax[0].set_title("Ego Pool — ELO per Model", fontsize=title_fontsize)
ax[0].set_ylabel("ELO", fontsize=label_fontsize)
ax[0].grid(True)
ax[0].legend(loc="upper left", fontsize=legend_fontsize)
padded_ylim_from_lines(ax[0])

# Row 2: NPC ELO
shade_by_cycle(ax[1], npc_spans, alpha=0.12)
plot_series(ax[1], npc_elo, npc_models, label_prefix="V")
ax[1].set_title("NPC Pool — ELO per Model", fontsize=title_fontsize)
ax[1].set_ylabel("ELO", fontsize=label_fontsize)
ax[1].grid(True)
ax[1].legend(loc="upper left", fontsize=legend_fontsize)
padded_ylim_from_lines(ax[1])

# Row 3: Ego & NPC variance on same axis

shade_by_cycle(ax[2], npc_spans, alpha=0.12)
ax[2].plot(ego_var, label="Ego Pool", linewidth=2)
ax[2].plot(npc_var, label="NPC Pool", linewidth=2)
ax[2].set_title("Probability Variance (vs Uniform)", fontsize=title_fontsize)
ax[2].set_ylabel("Variance", fontsize=label_fontsize)
ax[2].set_xlabel("Safety Hardening Cycle", fontsize=label_fontsize)
ax[2].grid(True)
ax[2].legend(loc="upper right", fontsize=legend_fontsize)

# Nice y-lims for variance
if np.isfinite(ego_var).any() or np.isfinite(npc_var).any():
    vmax = np.nanmax([np.nanmax(ego_var), np.nanmax(npc_var)])
    ax[2].set_ylim(0.0, vmax * 1.1 if vmax > 0 else 1.0)

# Shared x ticks (tweak as you like)
for a in ax:
    a.tick_params(axis='both', labelsize=tick_fontsize)
ax[2].set_xticks([0, 10, 30, 60, 100, 150])
ax[2].set_xticklabels(['1', '2', '3', '4', '5', ''])

ax[0].set_ylim(600, 1250)
ax[1].set_ylim(600, 1250)

ax[0].set_xlim(0, 150)
ax[1].set_xlim(0, 150)
ax[2].set_xlim(0, 150)

plt.tight_layout()
plt.savefig("elo_analysis.pdf")