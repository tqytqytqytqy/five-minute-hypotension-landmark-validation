#!/usr/bin/env python3
"""Generate publication figures from locked result-layer files only.

This script never fits a model and never writes patient-level predictions.  The
only individual-level input is the hashed U0 prediction file, which is first
verified against ``U0_result_SHA256SUMS.csv`` and then immediately aggregated
to precision-recall coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "05_MOVER_validation"
H5 = VALIDATION / "H5_primary"
OBS2 = VALIDATION / "observation_2x2"
R1 = VALIDATION / "R1_sensitivity"
DRIFT = VALIDATION / "subgroups_drift"
COHORT = ROOT / "03_derived_cohorts" / "MOVER"
FIG_DIR = ROOT / "07_figures"
SOURCE_DIR = FIG_DIR / "source_data"
QA_DIR = ROOT / "09_QA_reproducibility" / "reports"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#222222"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"
PALE_BLUE = "#DCEEF8"
PALE_ORANGE = "#FCE8C4"

MODEL_LABEL = {
    "LM5_common18": "LM5 common-18",
    "simple_recovered_by_5min": "Recovery by 5 min",
    "simple_early_mean_map": "Early mean MAP",
    "simple_t0_map": "t0 MAP",
}
MODEL_COLOR = {
    "LM5_common18": BLUE,
    "simple_recovered_by_5min": ORANGE,
    "simple_early_mean_map": GREEN,
    "simple_t0_map": GREY,
}
MODEL_STYLE = {
    "LM5_common18": "-",
    "simple_recovered_by_5min": "--",
    "simple_early_mean_map": ":",
    "simple_t0_map": "-.",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
        "patch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


MANIFEST_ROWS: list[dict[str, object]] = []
FIGURE_OUTPUTS: dict[str, tuple[Path, Path]] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def verify_u0_manifest() -> None:
    manifest = H5 / "U0_result_SHA256SUMS.csv"
    rows = pd.read_csv(manifest)
    required = {"file", "size_bytes", "sha256"}
    if not required.issubset(rows.columns):
        raise RuntimeError(f"Malformed U0 hash manifest: {manifest}")
    for row in rows.itertuples(index=False):
        path = H5 / str(row.file)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row.size_bytes):
            raise RuntimeError(f"U0 size mismatch: {path}")
        if sha256_file(path) != str(row.sha256):
            raise RuntimeError(f"U0 SHA256 mismatch: {path}")


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=BLACK,
    )


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, color=GREY)
    if grid:
        ax.grid(axis=grid, color=LIGHT_GREY, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    FIGURE_OUTPUTS[stem.split("_")[0]] = (png, pdf)
    return png, pdf


def write_source(
    figure_id: str,
    panel_id: str,
    name: str,
    frame: pd.DataFrame,
    upstream: Iterable[Path],
    hierarchy: str,
    notes: str,
) -> Path:
    if frame.empty:
        raise RuntimeError(f"Empty source data for {figure_id}{panel_id}: {name}")
    path = SOURCE_DIR / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    upstream_paths = [Path(p) for p in upstream]
    MANIFEST_ROWS.append(
        {
            "figure_id": figure_id,
            "panel_id": panel_id,
            "analysis_hierarchy": hierarchy,
            "source_data_file": relative(path),
            "source_data_sha256": sha256_file(path),
            "rows": len(frame),
            "columns": "|".join(map(str, frame.columns)),
            "upstream_files": "|".join(relative(p) for p in upstream_paths),
            "upstream_sha256": "|".join(f"{relative(p)}={sha256_file(p)}" for p in upstream_paths),
            "notes": notes,
        }
    )
    return path


def forest(
    ax: plt.Axes,
    data: pd.DataFrame,
    estimate: str,
    lower: str,
    upper: str,
    labels: list[str],
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    reference: float | None = None,
    xlabel: str = "",
    title: str = "",
) -> None:
    y = np.arange(len(data))[::-1]
    colors = colors or [BLUE] * len(data)
    markers = markers or ["o"] * len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        x = float(row[estimate])
        lo = float(row[lower])
        hi = float(row[upper])
        ax.errorbar(
            x,
            y[i],
            xerr=[[x - lo], [hi - x]],
            fmt=markers[i],
            color=colors[i],
            ecolor=colors[i],
            markersize=4.2,
            elinewidth=1.05,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    if reference is not None:
        ax.axvline(reference, color=GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    clean_axis(ax, "x")


def figure1() -> None:
    flow_path = COHORT / "mover_cohort_flow.csv"
    flow = read_csv(flow_path)
    wanted = [
        ("basic_common_operations", "Basic common operations"),
        ("H5_has_t0_after_3_to_30_min", "First H5 hypotension t0 at >3–30 min"),
        ("anesthesia_end_at_or_after_t0_plus_30", "Coverage through t0 + 30 min"),
        ("H5_feature_evaluable", "H5 feature evaluable"),
        ("fully_evaluable_t0_operations", "H5 feature + outcome evaluable"),
        ("first_fully_evaluable_operation_per_patient", "First evaluable operation per patient"),
    ]
    count = dict(zip(flow["step"], flow["n"]))
    source_rows = []
    for order, (step, label) in enumerate(wanted, start=1):
        source_rows.append(
            {
                "order": order,
                "step": step,
                "display_label": label,
                "n": int(count[step]),
                "excluded_from_previous": "" if order == 1 else int(source_rows[-1]["n"]) - int(count[step]),
                "branch": "screening",
            }
        )
    source_rows.extend(
        [
            {
                "order": 7,
                "step": "stage1_direct_alert_first_operations",
                "display_label": "Stage 1 direct alert",
                "n": int(count["stage1_direct_alert_first_operations"]),
                "excluded_from_previous": "",
                "branch": "stage1",
            },
            {
                "order": 7,
                "step": "main_stage2_model_cohort",
                "display_label": "Stage 2 frozen-model cohort",
                "n": int(count["main_stage2_model_cohort"]),
                "excluded_from_previous": "",
                "branch": "stage2",
            },
            {
                "order": 8,
                "step": "main_stage2_events",
                "display_label": "Stage 2 primary events",
                "n": int(count["main_stage2_events"]),
                "excluded_from_previous": "",
                "branch": "stage2_event",
            },
        ]
    )
    flow_source = pd.DataFrame(source_rows)
    write_source(
        "Fig1",
        "A",
        "Fig1A_cohort_flow",
        flow_source,
        [flow_path],
        "primary cohort construction",
        "Counts are copied from the locked MOVER cohort-flow result; exclusions are arithmetic differences.",
    )

    timeline = pd.DataFrame(
        [
            ("anaesthesia_start", -30, -3, np.nan, "Anaesthesia begins 3–30 min before t0", "eligibility", "t0 must occur >3 and <=30 min after start"),
            ("pre_t0_history", -10, 0, np.nan, "Pre-t0 MAP history", "features", "right-closed H5 source observations; no LOCF/interpolation"),
            ("t0", np.nan, np.nan, 0, "t0: first H5 MAP <65 mm Hg", "index", "after >3 and <=30 min"),
            ("early_window", 0, 5, np.nan, "Early trajectory (t0, t0+5]", "stage", "stage 1 if early recurrence or AUC65>=50"),
            ("landmark", np.nan, np.nan, 5, "Landmark t0+5", "prediction", "stage 2 frozen LM5 prediction"),
            ("outcome_window", 5, 30, np.nan, "Primary outcome (t0+5, t0+30]", "outcome", "recovery >10 min/never or AUC65>=75"),
        ],
        columns=["event_id", "start_min_relative_t0", "end_min_relative_t0", "marker_min_relative_t0", "label", "role", "locked_rule"],
    )
    sap = ROOT / "00_protocol_SAP" / "SAP_v1.0.md"
    endpoint = ROOT / "04_frozen_INSPIRE_LM5_model" / "cohort_endpoint.json"
    write_source(
        "Fig1",
        "B",
        "Fig1B_landmark_timeline",
        timeline,
        [sap, endpoint],
        "locked design schematic",
        "Relative-time schematic transcribed from the locked SAP and endpoint contract.",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 7.0), gridspec_kw={"width_ratios": [0.95, 1.15]})
    add_panel_label(ax1, "A")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    y_positions = [0.95, 0.81, 0.67, 0.53, 0.39, 0.25]
    box_w, box_h = 0.72, 0.073
    for i, ((_, label), y) in enumerate(zip(wanted, y_positions)):
        n = int(flow_source.iloc[i]["n"])
        box = FancyBboxPatch(
            (0.5 - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            facecolor=PALE_BLUE if i < 5 else "#E8F3E8",
            edgecolor=BLUE if i < 5 else GREEN,
            linewidth=0.9,
        )
        ax1.add_patch(box)
        ax1.text(0.5, y, f"{label}\nn={n:,}", ha="center", va="center", fontsize=6.8)
        if i < len(y_positions) - 1:
            next_y = y_positions[i + 1]
            ax1.add_patch(
                FancyArrowPatch(
                    (0.5, y - box_h / 2),
                    (0.5, next_y + box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.7,
                    color=GREY,
                )
            )
            excluded = int(flow_source.iloc[i + 1]["excluded_from_previous"])
            ax1.text(0.88, (y + next_y) / 2, f"Excluded\nn={excluded:,}", ha="right", va="center", fontsize=5.8, color=GREY)
    branch_y = 0.075
    branch_info = [
        (0.27, "Stage 1 direct alert", int(count["stage1_direct_alert_first_operations"]), ORANGE),
        (0.73, "Stage 2 frozen model", int(count["main_stage2_model_cohort"]), BLUE),
    ]
    for x, label, n, color in branch_info:
        ax1.add_patch(
            FancyBboxPatch(
                (x - 0.185, branch_y - 0.047),
                0.37,
                0.094,
                boxstyle="round,pad=0.01,rounding_size=0.012",
                facecolor=PALE_ORANGE if color == ORANGE else PALE_BLUE,
                edgecolor=color,
                linewidth=0.9,
            )
        )
        event_text = f"; events={int(count['main_stage2_events']):,}" if "Stage 2" in label else ""
        ax1.text(x, branch_y, f"{label}\nn={n:,}{event_text}", ha="center", va="center", fontsize=6.5)
        ax1.add_patch(
            FancyArrowPatch(
                (0.5, y_positions[-1] - box_h / 2),
                (x, branch_y + 0.047),
                arrowstyle="-|>",
                mutation_scale=8,
                connectionstyle=f"arc3,rad={-0.12 if x < 0.5 else 0.12}",
                linewidth=0.7,
                color=GREY,
            )
        )
    ax1.set_title("MOVER external-validation cohort", loc="left", fontweight="bold", pad=8)

    add_panel_label(ax2, "B")
    ax2.set_xlim(-12, 32)
    ax2.set_ylim(0, 1)
    ax2.spines[["left", "right", "top"]].set_visible(False)
    ax2.set_yticks([])
    ax2.set_xticks([-10, -5, 0, 5, 10, 15, 20, 25, 30])
    ax2.set_xlabel("Minutes relative to t0")
    ax2.axhline(0.50, color=BLACK, linewidth=0.8)
    for x in range(-10, 31, 5):
        ax2.plot(x, 0.50, "o", color=BLUE if x <= 5 else GREY, markersize=3.5, zorder=3)
    ax2.annotate("Anaesthesia start\n3–30 min before t0", xy=(0, 0.91), xytext=(-10.5, 0.91), arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8), ha="center", va="center", fontsize=6.6)
    ax2.plot([-10, 0], [0.72, 0.72], color=BLUE, linewidth=7, solid_capstyle="butt")
    ax2.text(-5, 0.77, "Pre-t0 MAP history", ha="center", va="bottom", fontsize=7)
    ax2.plot([0, 5], [0.62, 0.62], color=ORANGE, linewidth=10, solid_capstyle="butt")
    ax2.text(2.5, 0.68, "Early trajectory", ha="center", va="bottom", fontsize=7)
    ax2.text(2.5, 0.57, "Stage 1 if recurrence / AUC65≥50", ha="center", va="top", fontsize=5.9)
    ax2.plot([5, 30], [0.32, 0.32], color=PURPLE, linewidth=10, solid_capstyle="butt")
    ax2.text(17.5, 0.38, "Primary outcome window", ha="center", va="bottom", fontsize=7)
    ax2.text(17.5, 0.27, "Recovery >10 min/never or AUC65≥75", ha="center", va="top", fontsize=5.9)
    ax2.axvline(0, color=VERMILION, linewidth=1.0)
    ax2.text(-0.7, 0.98, "t0\nfirst H5 MAP <65", ha="right", va="top", fontsize=6.5, color=VERMILION)
    ax2.axvline(5, color=BLUE, linewidth=1.0, linestyle="--")
    ax2.text(5.7, 0.98, "t0+5 landmark\nStage 2 prediction", ha="left", va="top", fontsize=6.5, color=BLUE)
    ax2.text(10, 0.08, "H5 source observations only; no LOCF or interpolation", ha="center", va="center", fontsize=6.2, color=GREY)
    ax2.set_title("Landmark and outcome timing", loc="left", fontweight="bold", pad=8)
    fig.subplots_adjust(wspace=0.32, left=0.05, right=0.98, top=0.95, bottom=0.08)
    save_figure(fig, "Fig1_cohort_flow_and_landmark_timeline")


def figure2() -> None:
    rcs_path = H5 / "U0_primary_RCS_calibration_curve_with_95CI.csv"
    bins_path = H5 / "U0_calibration_equal_frequency_bins.csv"
    pred_path = H5 / "U0_hashed_individual_predictions.csv.gz"
    point_path = H5 / "U0_external_validation_point_estimates.csv"
    dca_path = H5 / "U0_decision_curve_0.05_0.50.csv"
    trade_path = H5 / "U0_two_stage_fixed_strategy_bootstrap_2000_summary.csv"

    rcs = read_csv(rcs_path)
    bins = read_csv(bins_path).query("model == 'LM5_common18'").copy()
    write_source("Fig2", "A", "Fig2A_calibration_RCS", rcs, [rcs_path], "primary H5/U0", "Locked RCS calibration curve with patient-bootstrap 95% CI.")
    write_source("Fig2", "A", "Fig2A_calibration_bins", bins, [bins_path], "primary H5/U0", "Equal-frequency bins for LM5 only.")

    pred = read_csv(pred_path)
    if any(c.endswith("_hash") for c in pred.columns if c not in {"patient_hash", "case_hash"}):
        raise RuntimeError("Unexpected hashed identifier columns")
    y = pred["primary_outcome"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Primary outcome is not binary")
    points = read_csv(point_path)
    pr_rows = []
    pred_cols = {
        "LM5_common18": "probability_LM5_common18",
        "simple_recovered_by_5min": "probability_simple_recovered_by_5min",
        "simple_early_mean_map": "probability_simple_early_mean_map",
        "simple_t0_map": "probability_simple_t0_map",
    }
    for model, col in pred_cols.items():
        p = pred[col].to_numpy(dtype=float)
        precision, recall, threshold = precision_recall_curve(y, p)
        ap = float(average_precision_score(y, p))
        locked_ap = float(points.loc[points["model"] == model, "auprc"].iloc[0])
        if not math.isclose(ap, locked_ap, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"AUPRC mismatch for {model}: {ap} vs {locked_ap}")
        threshold_full = np.append(threshold, np.nan)
        pr_rows.extend(
            {
                "model": model,
                "recall": float(r),
                "precision": float(q),
                "threshold": float(t) if np.isfinite(t) else np.nan,
                "locked_auprc": locked_ap,
                "event_rate": float(y.mean()),
            }
            for r, q, t in zip(recall, precision, threshold_full)
        )
    pr = pd.DataFrame(pr_rows)
    write_source(
        "Fig2",
        "B",
        "Fig2B_precision_recall_curve",
        pr,
        [pred_path, point_path, H5 / "U0_result_SHA256SUMS.csv"],
        "primary H5/U0",
        "Only aggregate PR coordinates are exported; patient_hash and case_hash are never written to figure source data.",
    )

    dca = read_csv(dca_path)
    write_source("Fig2", "C", "Fig2C_decision_curve", dca, [dca_path], "primary H5/U0", "Locked DCA over 0.05–0.50; treat-all and treat-none are reference strategies.")

    trade = read_csv(trade_path)
    key_cols = ["threshold", "threshold_name", "threshold_type", "source"]
    selected_metrics = ["alerts_per_1000", "sensitivity", "positive_predictive_value", "alerts_per_true_positive"]
    wide_parts = []
    for metric in selected_metrics:
        part = trade.loc[trade["metric"] == metric, key_cols + ["estimate", "ci95_lower", "ci95_upper"]].copy()
        part = part.rename(columns={"estimate": metric, "ci95_lower": f"{metric}_ci95_lower", "ci95_upper": f"{metric}_ci95_upper"})
        wide_parts.append(part)
    trade_wide = wide_parts[0]
    for part in wide_parts[1:]:
        trade_wide = trade_wide.merge(part, on=key_cols, how="outer", validate="one_to_one")
    trade_wide = trade_wide.sort_values("threshold").reset_index(drop=True)
    write_source("Fig2", "D", "Fig2D_two_stage_threshold_tradeoff", trade_wide, [trade_path], "primary two-stage H5/U0", "Stage 1 direct alerts plus Stage 2 frozen-model alerts; all thresholds were locked before MOVER performance review.")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6))
    ax = axs[0, 0]
    add_panel_label(ax, "A")
    ax.fill_between(rcs["predicted_probability"], rcs["ci95_lower"], rcs["ci95_upper"], color=SKY, alpha=0.25, linewidth=0)
    ax.plot(rcs["predicted_probability"], rcs["rcs_calibrated_observed_probability"], color=BLUE, label="RCS calibration")
    ax.errorbar(bins["mean_predicted"], bins["observed_rate"], yerr=np.sqrt(bins["observed_rate"] * (1 - bins["observed_rate"]) / bins["n"]) * 1.96, fmt="o", color=VERMILION, markersize=3.5, capsize=1.8, label="Equal-frequency bins")
    lim = 0.65
    ax.plot([0, lim], [0, lim], linestyle="--", color=GREY, linewidth=0.8, label="Ideal")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted probability", ylabel="Observed probability")
    ax.set_title("Calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax, "both")

    ax = axs[0, 1]
    add_panel_label(ax, "B")
    for model in MODEL_LABEL:
        d = pr.loc[pr["model"] == model]
        lw = 1.7 if model == "LM5_common18" else 1.0
        ax.plot(d["recall"], d["precision"], color=MODEL_COLOR[model], linestyle=MODEL_STYLE[model], linewidth=lw, label=f"{MODEL_LABEL[model]} ({d['locked_auprc'].iloc[0]:.3f})")
    ax.axhline(float(y.mean()), color=BLACK, linestyle="--", linewidth=0.8, label=f"Event rate ({y.mean():.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall (sensitivity)", ylabel="Precision (positive predictive value)")
    ax.set_title("Precision–recall curve (AUPRC)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax, "both")

    ax = axs[1, 0]
    add_panel_label(ax, "C")
    for model in MODEL_LABEL:
        d = dca.loc[dca["model"] == model]
        lw = 1.7 if model == "LM5_common18" else 0.95
        ax.plot(d["threshold"], d["net_benefit_model"], color=MODEL_COLOR[model], linestyle=MODEL_STYLE[model], linewidth=lw, label=MODEL_LABEL[model])
    ref = dca.loc[dca["model"] == "LM5_common18"]
    ax.plot(ref["threshold"], ref["net_benefit_all"], color=BLACK, linestyle="--", linewidth=0.9, label="Treat all")
    ax.plot(ref["threshold"], ref["net_benefit_none"], color=GREY, linestyle=":", linewidth=0.9, label="Treat none")
    for t in [0.10, 0.15, 0.20, 0.25]:
        ax.axvline(t, color=LIGHT_GREY, linewidth=0.55, zorder=0)
    ax.set(xlim=(0.05, 0.50), ylim=(-0.05, 0.17), xlabel="Risk threshold", ylabel="Net benefit")
    ax.set_title("Decision-curve analysis", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    clean_axis(ax, "both")

    ax = axs[1, 1]
    add_panel_label(ax, "D")
    clinical = trade_wide["threshold_type"].str.startswith("prespecified_clinical")
    colors = np.where(clinical, ORANGE, BLUE)
    markers = np.where(clinical, "o", "s")
    short_labels = {
        "clinical_action_sensitivity_0.10": "C10",
        "clinical_action_primary": "C15*",
        "clinical_action_sensitivity_0.20": "C20",
        "clinical_action_sensitivity_0.25": "C25",
        "capacity_top_30_percent": "Top30",
        "capacity_top_20_percent": "Top20",
        "capacity_top_10_percent": "Top10",
    }
    for i, row in trade_wide.iterrows():
        x = float(row["alerts_per_1000"])
        yy = float(row["sensitivity"])
        ax.errorbar(
            x,
            yy,
            xerr=[[x - float(row["alerts_per_1000_ci95_lower"])], [float(row["alerts_per_1000_ci95_upper"]) - x]],
            yerr=[[yy - float(row["sensitivity_ci95_lower"])], [float(row["sensitivity_ci95_upper"]) - yy]],
            fmt=markers[i],
            color=colors[i],
            ecolor=colors[i],
            markersize=4.5,
            capsize=1.7,
            markeredgecolor="white",
            markeredgewidth=0.45,
        )
        offset = (4, 4) if i % 2 == 0 else (4, -8)
        ax.annotate(short_labels.get(row["threshold_name"], f"{row['threshold']:.2f}"), (x, yy), xytext=offset, textcoords="offset points", fontsize=6)
    ax.scatter([], [], marker="o", color=ORANGE, label="Clinical threshold")
    ax.scatter([], [], marker="s", color=BLUE, label="INSPIRE capacity threshold")
    ax.set(xlabel="Alerts per 1,000 landmarks", ylabel="Event sensitivity", ylim=(0.45, 0.94))
    ax.set_title("Two-stage alert trade-off", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax, "both")
    fig.text(0.99, 0.005, "C15* = prespecified primary clinical threshold; bars are 95% patient-bootstrap CIs.", ha="right", va="bottom", fontsize=6.2, color=GREY)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.10, wspace=0.28, hspace=0.32)
    save_figure(fig, "Fig2_primary_external_validation")


def figure3() -> None:
    factorial_path = OBS2 / "observation_2x2_complete_factorial_bootstrap_2000.csv"
    raw = read_csv(factorial_path)
    cells = raw.loc[(raw["estimand_kind"] == "cell") & raw["estimand"].str.endswith(("__auroc", "__auprc"))].copy()
    cells["metric"] = cells["estimand"].str.rsplit("__", n=1).str[-1]
    cells["cell"] = cells["estimand"].str.rsplit("__", n=1).str[0]
    cell_label = {
        "H5_features__H5_outcome": "H5 features / H5 outcome",
        "H5_features__R1_outcome": "H5 features / R1 outcome",
        "R1_features__H5_outcome": "R1 features / H5 outcome",
        "R1_features__R1_outcome": "R1 features / R1 outcome",
    }
    cells["display_label"] = cells["cell"].map(cell_label)
    effects = raw.loc[(raw["estimand_kind"] == "factorial_effect") & raw["estimand"].str.contains("marginal_feature_R1_minus_H5|marginal_outcome_R1_minus_H5|feature_by_outcome_interaction", regex=True) & raw["estimand"].str.endswith(("__auroc", "__auprc"))].copy()
    effects["metric"] = effects["estimand"].str.rsplit("__", n=1).str[-1]
    def effect_label(x: str) -> str:
        if "marginal_feature" in x:
            return "Feature process: R1 − H5"
        if "marginal_outcome" in x:
            return "Outcome process: R1 − H5"
        return "Feature × outcome interaction"
    effects["display_label"] = effects["estimand"].map(effect_label)
    source = pd.concat([cells, effects], ignore_index=True)
    write_source("Fig3", "A-D", "Fig3_H5_R1_2x2_factorial_forest", source, [factorial_path], "prespecified observation-process sensitivity", "Complete four-cell intersection; patient-level paired bootstrap. Negative AUROC/AUPRC factorial effects favor H5 over R1.")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.2))
    order_cells = ["H5 features / H5 outcome", "H5 features / R1 outcome", "R1 features / H5 outcome", "R1 features / R1 outcome"]
    for ax, metric, panel, title, xlabel in [
        (axs[0, 0], "auroc", "A", "Four observation-process cells", "AUROC"),
        (axs[0, 1], "auprc", "B", "Four observation-process cells", "AUPRC"),
    ]:
        d = cells.loc[cells["metric"] == metric].set_index("display_label").loc[order_cells].reset_index()
        colors = [BLUE, BLUE, ORANGE, ORANGE]
        markers = ["o", "s", "o", "s"]
        forest(ax, d, "estimate", "ci95_lower", "ci95_upper", order_cells, colors, markers, xlabel=xlabel, title=title)
        add_panel_label(ax, panel)
        if panel == "B":
            ax.tick_params(axis="y", labelleft=False)
    order_eff = ["Feature process: R1 − H5", "Outcome process: R1 − H5", "Feature × outcome interaction"]
    for ax, metric, panel, xlabel in [
        (axs[1, 0], "auroc", "C", "Difference in AUROC"),
        (axs[1, 1], "auprc", "D", "Difference in AUPRC"),
    ]:
        d = effects.loc[effects["metric"] == metric].set_index("display_label").loc[order_eff].reset_index()
        forest(ax, d, "estimate", "ci95_lower", "ci95_upper", order_eff, [GREEN, PURPLE, GREY], ["o", "s", "D"], reference=0, xlabel=xlabel, title="Factorial effects")
        add_panel_label(ax, panel)
        if panel == "D":
            ax.tick_params(axis="y", labelleft=False)
    fig.text(0.99, 0.005, "Complete intersection n=1,488. H5: 5-min process; R1: 1-min process. Error bars are 95% patient-bootstrap CIs.", ha="right", va="bottom", fontsize=6.2, color=GREY)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.95, bottom=0.11, wspace=0.38, hspace=0.38)
    save_figure(fig, "Fig3_observation_process_2x2_factorial")


def supplement_phase_operator() -> None:
    path = R1 / "phase_buffer_R1_redetect_performance.csv"
    raw = read_csv(path)
    label_map = {
        ("H5_phase", "0"): "H5 primary (phase 0)",
        ("H5_phase", "1"): "H5 phase offset 1",
        ("H5_phase", "2"): "H5 phase offset 2",
        ("H5_phase", "3"): "H5 phase offset 3",
        ("H5_phase", "4"): "H5 phase offset 4",
        ("induction_buffer_exclusive_min", "5.0"): "Exclude first 5 min",
        ("induction_buffer_exclusive_min", "10.0"): "Exclude first 10 min",
        ("H5_operator_variant", "right_closed_nearest_right_boundary_ART_priority"): "Nearest right boundary; ART priority",
        ("H5_operator_variant", "right_closed_median_NIBP_priority"): "Median; NIBP priority",
        ("H5_operator_variant", "left_closed_median_ART_priority"): "Left-closed; median; ART priority",
        ("R1_redetected_t0", "0"): "R1 re-detected t0",
    }
    raw["setting_key"] = raw["setting"].astype(str)
    raw["display_label"] = [label_map.get((a, s), f"{a}: {s}") for a, s in zip(raw["analysis"], raw["setting_key"])]
    raw["display_label_n"] = raw["display_label"] + " (n=" + raw["n"].map(lambda x: f"{int(x):,}") + ")"
    cols = ["analysis", "setting", "display_label", "n", "events", "event_rate", "auroc", "auroc_patient_bootstrap_ci_lower", "auroc_patient_bootstrap_ci_upper", "auprc", "auprc_patient_bootstrap_ci_lower", "auprc_patient_bootstrap_ci_upper", "estimability_status", "nonestimable_reason"]
    source = raw[cols]
    write_source("FigS1", "A-B", "FigS1_phase_operator_R1_sensitivity", source, [path], "secondary observation-process sensitivity", "Full cohort re-selection under each observation variant; H5 phase 0 is the primary reference.")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 5.0))
    labels = raw["display_label_n"].tolist()
    colors = [BLUE if x == "H5 primary (phase 0)" else (ORANGE if x == "R1 re-detected t0" else GREY) for x in raw["display_label"]]
    forest(axs[0], raw, "auroc", "auroc_patient_bootstrap_ci_lower", "auroc_patient_bootstrap_ci_upper", labels, colors, xlabel="AUROC", title="Discrimination")
    add_panel_label(axs[0], "A")
    forest(axs[1], raw, "auprc", "auprc_patient_bootstrap_ci_lower", "auprc_patient_bootstrap_ci_upper", labels, colors, xlabel="AUPRC", title="Precision–recall performance")
    add_panel_label(axs[1], "B")
    axs[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.38, right=0.98, top=0.93, bottom=0.12, wspace=0.35)
    save_figure(fig, "FigS1_phase_operator_R1_sensitivity")


def supplement_endpoints() -> None:
    perf_path = R1 / "endpoint_and_coverage_sensitivity_performance.csv"
    boot_path = R1 / "endpoint_and_coverage_patient_bootstrap_2000.csv"
    primary_point_path = H5 / "U0_external_validation_point_estimates.csv"
    primary_boot_path = H5 / "U0_paired_patient_bootstrap_2000_summary.csv"
    perf = read_csv(perf_path)
    boot = read_csv(boot_path)
    point = read_csv(primary_point_path).query("model == 'LM5_common18'").iloc[0]
    pboot = read_csv(primary_boot_path)
    endpoint_label = {
        "primary": "Primary: persistent >10 min or AUC65≥75",
        "persistent_15_or_auc75": "Persistent >15 min or AUC65≥75",
        "persistent_10_or_auc50": "Persistent >10 min or AUC65≥50",
        "MAP60_persistent10_or_auc75": "MAP<60: persistent >10 min or AUC65≥75",
        "H5_all_6_points_max_gap_5": "Complete H5 coverage (6/6; gap≤5 min)",
    }
    rows = []
    metrics = ["auroc", "auprc", "brier", "calibration_slope"]
    for metric in metrics:
        b = pboot.loc[pboot["estimand"] == f"LM5_common18__{metric}"].iloc[0]
        rows.append({"endpoint": "primary", "display_label": endpoint_label["primary"], "metric": metric, "estimate": float(point[metric]), "ci_lower": float(b["ci95_lower"]), "ci_upper": float(b["ci95_upper"]), "n": int(point["n"]), "events": int(point["events"])})
    for ep, group in boot.groupby("endpoint_sensitivity", sort=False):
        prow = perf.loc[perf["endpoint_sensitivity"] == ep].iloc[0]
        for metric in metrics:
            b = group.loc[group["metric"] == metric].iloc[0]
            rows.append({"endpoint": ep, "display_label": endpoint_label.get(ep, ep), "metric": metric, "estimate": float(b["estimate"]), "ci_lower": float(b["ci_lower"]), "ci_upper": float(b["ci_upper"]), "n": int(prow["n"]), "events": int(prow["events"])})
    source = pd.DataFrame(rows)
    write_source("FigS2", "A-D", "FigS2_endpoint_coverage_sensitivity", source, [perf_path, boot_path, primary_point_path, primary_boot_path], "secondary endpoint sensitivity", "Primary endpoint added as reference; all uncertainty is patient-bootstrap 95% CI.")
    order = list(endpoint_label)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.1))
    spec = [("auroc", "A", "AUROC", None), ("auprc", "B", "AUPRC", None), ("brier", "C", "Brier score (lower is better)", None), ("calibration_slope", "D", "Calibration slope", 1.0)]
    for ax, (metric, panel, xlabel, ref) in zip(axs.flat, spec):
        d = source.loc[source["metric"] == metric].set_index("endpoint").loc[order].reset_index()
        labels = [f"{r.display_label} (n={int(r.n):,})" for r in d.itertuples()]
        colors = [BLUE] + [ORANGE, GREEN, PURPLE, GREY]
        forest(ax, d, "estimate", "ci_lower", "ci_upper", labels, colors, reference=ref, xlabel=xlabel, title=xlabel)
        add_panel_label(ax, panel)
        if panel in {"B", "D"}:
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.42, right=0.98, top=0.95, bottom=0.11, wspace=0.42, hspace=0.37)
    save_figure(fig, "FigS2_endpoint_coverage_sensitivity")


def supplement_subgroups() -> None:
    path = DRIFT / "prespecified_subgroup_performance.csv"
    raw = read_csv(path)
    nice = {
        "age_lt65": "Age <65 y",
        "age_ge65": "Age ≥65 y",
        "female": "Female",
        "male": "Male",
        "ASA_1_2": "ASA I–II",
        "ASA_ge3": "ASA ≥III",
        "BMI_lt25": "BMI <25",
        "BMI_25_30": "BMI 25–<30",
        "BMI_ge30": "BMI ≥30",
        "t0_ART": "t0 arterial",
        "t0_NIBP": "t0 non-invasive",
        "t0_early_le10": "t0 ≤10 min",
        "t0_late_gt10": "t0 >10 min",
    }
    raw["display_label"] = raw["subgroup"].map(nice).fillna(raw["subgroup"])
    raw["display_label_n"] = [f"{lab} (n={int(n):,}; {100*ev/n:.1f}% events)" for lab, n, ev in zip(raw["display_label"], raw["n"], raw["events"])]
    cols = ["subgroup", "display_label", "n", "events", "event_rate", "auroc", "auroc_patient_bootstrap_ci_lower", "auroc_patient_bootstrap_ci_upper", "calibration_slope", "calibration_slope_patient_bootstrap_ci_lower", "calibration_slope_patient_bootstrap_ci_upper", "estimability_status", "nonestimable_reason"]
    source = raw[cols]
    write_source("FigS3", "A-B", "FigS3_prespecified_subgroup_forest", source, [path], "secondary prespecified subgroup analysis", "All prespecified subgroups retained; no subgroup was silently dropped.")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 6.0))
    labels = raw["display_label_n"].tolist()
    forest(axs[0], raw, "auroc", "auroc_patient_bootstrap_ci_lower", "auroc_patient_bootstrap_ci_upper", labels, [BLUE] * len(raw), xlabel="AUROC", title="Discrimination")
    add_panel_label(axs[0], "A")
    forest(axs[1], raw, "calibration_slope", "calibration_slope_patient_bootstrap_ci_lower", "calibration_slope_patient_bootstrap_ci_upper", labels, [ORANGE] * len(raw), reference=1.0, xlabel="Calibration slope", title="Calibration")
    add_panel_label(axs[1], "B")
    axs[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.39, right=0.98, top=0.94, bottom=0.10, wspace=0.40)
    save_figure(fig, "FigS3_prespecified_subgroup_forest")


def supplement_drift() -> None:
    path = DRIFT / "feature_drift_INSPIRE_vs_MOVER.csv"
    raw = read_csv(path)
    nice = {
        "age_years": "Age",
        "male": "Male",
        "bmi": "BMI",
        "asa": "ASA",
        "t0_map": "t0 MAP",
        "t0_map_squared": "t0 MAP²",
        "t0_arterial_source": "t0 arterial source",
        "anesthesia_start_to_t0_min": "Start-to-t0 time",
        "pre10_map_record_count": "Pre-t0 MAP count",
        "pre10_last_measurement_gap_min": "Pre-t0 last gap",
        "pre10_last_map": "Pre-t0 last MAP",
        "pre10_mean_map": "Pre-t0 mean MAP",
        "pre10_map_ols_slope_per_min": "Pre-t0 MAP slope",
        "recovered_by_5min": "Recovered by 5 min",
        "early_auc65_0_5_mmhg_min": "Early AUC65",
        "early_min_map_0_5": "Early minimum MAP",
        "early_mean_map_0_5": "Early mean MAP",
        "early_map_record_count_0_5": "Early MAP count",
    }
    raw["display_label"] = raw["feature"].map(nice).fillna(raw["feature"])
    raw["smd_estimability_status"] = np.where(raw["standardized_mean_difference_MOVER_minus_INSPIRE"].notna(), "estimable", "nonestimable_zero_pooled_variance")
    write_source("FigS4", "A-B", "FigS4_feature_drift", raw, [path], "secondary transportability/drift analysis", "Out-of-range denominator is nonmissing MOVER observations. NaN SMD is retained and marked non-estimable.")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 6.1))
    y = np.arange(len(raw))[::-1]
    ax = axs[0]
    add_panel_label(ax, "A")
    smd = raw["standardized_mean_difference_MOVER_minus_INSPIRE"].to_numpy(float)
    for i, val in enumerate(smd):
        if np.isfinite(val):
            color = VERMILION if abs(val) >= 0.5 else (ORANGE if abs(val) >= 0.2 else BLUE)
            ax.plot([0, val], [y[i], y[i]], color=color, linewidth=1.0)
            ax.plot(val, y[i], "o", color=color, markersize=4)
        else:
            ax.plot(0, y[i], marker="x", color=GREY, markersize=4)
            ax.text(0.03, y[i], "NE", va="center", fontsize=5.8, color=GREY)
    ax.axvline(0, color=BLACK, linewidth=0.75)
    for v in [-0.5, -0.2, 0.2, 0.5]:
        ax.axvline(v, color=LIGHT_GREY, linewidth=0.65, linestyle="--")
    ax.set_yticks(y, raw["display_label"])
    ax.set_xlabel("Standardized mean difference\n(MOVER − INSPIRE)")
    ax.set_title("Feature distribution shift", loc="left", fontweight="bold")
    clean_axis(ax, "x")
    ax = axs[1]
    add_panel_label(ax, "B")
    ax.scatter(raw["MOVER_missing_percent"], y + 0.12, marker="o", color=ORANGE, s=17, label="Missing")
    ax.scatter(raw["MOVER_outside_INSPIRE_p01_p99_percent"], y - 0.12, marker="s", color=BLUE, s=16, label="Outside INSPIRE P1–P99")
    ax.set_yticks(y, raw["display_label"])
    ax.set_xlabel("MOVER observations (%)")
    ax.set_title("Missingness and range transport", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.tick_params(axis="y", labelleft=False)
    clean_axis(ax, "x")
    fig.subplots_adjust(left=0.32, right=0.98, top=0.94, bottom=0.12, wspace=0.38)
    save_figure(fig, "FigS4_feature_drift")


def supplement_updates() -> None:
    eval_path = DRIFT / "U1_U2_independent_update_evaluation.csv"
    boot_path = DRIFT / "U1_U2_paired_bootstrap_2000.csv"
    eval_df = read_csv(eval_path)
    boot = read_csv(boot_path)
    states = ["U0", "U1_intercept", "U2_intercept_slope"]
    labels = {"U0": "U0 frozen", "U1_intercept": "U1 intercept", "U2_intercept_slope": "U2 intercept+slope"}
    metrics = ["brier", "log_loss", "calibration_slope", "calibration_in_the_large"]
    source = boot.loc[(boot["kind"] == "model") & boot["estimand"].isin(states) & boot["metric"].isin(metrics)].copy()
    source["display_label"] = source["estimand"].map(labels)
    source = source.merge(eval_df[["model_state", "update_n", "evaluation_n", "update_events", "evaluation_events", "validity_status"]], left_on="estimand", right_on="model_state", how="left", validate="many_to_one")
    write_source("FigS5", "A-D", "FigS5_U1_U2_update_evaluation", source, [eval_path, boot_path], "secondary independent local-update analysis", "Updates were fitted in one patient half and evaluated in the independent other half; paired patient-bootstrap CIs shown.")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    spec = [
        ("brier", "A", "Brier score", None),
        ("log_loss", "B", "Log loss", None),
        ("calibration_slope", "C", "Calibration slope", 1.0),
        ("calibration_in_the_large", "D", "Calibration-in-the-large", 0.0),
    ]
    x = np.arange(3)
    for ax, (metric, panel, title, ref) in zip(axs.flat, spec):
        d = source.loc[source["metric"] == metric].set_index("estimand").loc[states].reset_index()
        ax.errorbar(x, d["estimate"], yerr=[d["estimate"] - d["ci_lower"], d["ci_upper"] - d["estimate"]], fmt="o-", color=BLUE, ecolor=BLUE, capsize=2.2, markersize=4.5)
        if ref is not None:
            ax.axhline(ref, color=GREY, linestyle="--", linewidth=0.8)
        ax.set_xticks(x, [labels[s] for s in states], rotation=18, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title, loc="left", fontweight="bold")
        add_panel_label(ax, panel)
        if metric == "auroc":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        elif metric in {"auprc", "brier"}:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        clean_axis(ax, "y")
    fig.text(0.99, 0.005, "Independent evaluation half n=3,669; update half n=3,508. Error bars are 95% patient-bootstrap CIs.", ha="right", fontsize=6.2, color=GREY)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.16, wspace=0.28, hspace=0.38)
    save_figure(fig, "FigS5_U1_U2_update_evaluation")


def supplement_ipw() -> None:
    path = DRIFT / "outcome_observability_IPW_sensitivity.csv"
    raw = read_csv(path)
    raw["display_label"] = raw["analysis"].map({"unweighted_complete_outcome": "Complete outcome", "IPW_1st_99th_truncated": "IPW truncated P1–P99"})
    write_source("FigS6", "A-D", "FigS6_outcome_observability_IPW", raw, [path], "secondary outcome-observability sensitivity", "Protocol >20% missing-outcome gate was not triggered; point estimates only because this result file contains no bootstrap CI.")
    metrics = [("auroc", "A", "AUROC", None), ("auprc", "B", "AUPRC", None), ("brier", "C", "Brier score", None), ("calibration_slope", "D", "Calibration slope", 1.0)]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, (metric, panel, title, ref) in zip(axs.flat, metrics):
        vals = raw[metric].to_numpy(float)
        ax.plot([0, 1], vals, color=GREY, linewidth=1.0)
        ax.scatter([0, 1], vals, c=[BLUE, ORANGE], s=28, zorder=3, edgecolor="white", linewidth=0.5)
        if ref is not None:
            ax.axhline(ref, color=GREY, linestyle="--", linewidth=0.8)
        ax.set_xticks([0, 1], raw["display_label"], rotation=12, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title, loc="left", fontweight="bold")
        add_panel_label(ax, panel)
        if metric == "auroc":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        elif metric in {"auprc", "brier"}:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        clean_axis(ax, "y")
    miss = float(raw["outcome_unobserved_percent"].iloc[0])
    ess = float(raw.loc[raw["analysis"] == "IPW_1st_99th_truncated", "effective_sample_size"].iloc[0])
    fig.text(0.99, 0.005, f"Outcome unobserved: {miss:.1f}%; IPW effective sample size: {ess:,.1f}. Point estimates only (no bootstrap CI in locked result).", ha="right", fontsize=6.2, color=GREY)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.18, wspace=0.28, hspace=0.38)
    save_figure(fig, "FigS6_outcome_observability_IPW")


def supplement_vasopressor() -> None:
    perf_path = DRIFT / "early_vasopressor_descriptive_performance.csv"
    boot_path = DRIFT / "early_vasopressor_patient_bootstrap_2000.csv"
    perf = read_csv(perf_path)
    boot = read_csv(boot_path)
    estimable = perf.loc[perf["estimability_status"] == "estimable"].copy()
    labels = {
        "confirmed_new_bolus_or_rate_change_0_5": "Confirmed new bolus/rate change",
        "rate_verify_only_without_confirmed_action_0_5": "Rate verify only",
        "no_candidate_vasopressor_action_recorded_0_5": "No candidate action recorded",
    }
    rows = []
    for prow in perf.itertuples(index=False):
        stratum = prow.treatment_stratum
        for metric in ["auroc", "auprc", "calibration_slope"]:
            b = boot.loc[(boot["treatment_stratum"] == stratum) & (boot["metric"] == metric)]
            rows.append(
                {
                    "treatment_stratum": stratum,
                    "display_label": labels.get(stratum, stratum),
                    "n": int(prow.n),
                    "events": int(prow.events),
                    "event_rate": prow.event_rate,
                    "estimability_status": prow.estimability_status,
                    "nonestimable_reason": prow.nonestimable_reason,
                    "metric": metric,
                    "estimate": float(b["estimate"].iloc[0]) if len(b) else np.nan,
                    "ci_lower": float(b["ci_lower"].iloc[0]) if len(b) else np.nan,
                    "ci_upper": float(b["ci_upper"].iloc[0]) if len(b) else np.nan,
                }
            )
    source = pd.DataFrame(rows)
    write_source("FigS7", "A-C", "FigS7_early_vasopressor_descriptive", source, [perf_path, boot_path], "secondary descriptive treatment-stratified analysis", "Recorded treatment only; never causal. Non-estimable rate-verify-only stratum retained explicitly.")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1))
    spec = [("auroc", "A", "AUROC", None), ("auprc", "B", "AUPRC", None), ("calibration_slope", "C", "Calibration slope", 1.0)]
    order = ["confirmed_new_bolus_or_rate_change_0_5", "no_candidate_vasopressor_action_recorded_0_5"]
    for ax, (metric, panel, title, ref) in zip(axs, spec):
        d = source.loc[(source["metric"] == metric) & (source["estimability_status"] == "estimable")].set_index("treatment_stratum").loc[order].reset_index()
        ylabels = [f"{r.display_label}\n(n={int(r.n):,})" for r in d.itertuples()]
        forest(ax, d, "estimate", "ci_lower", "ci_upper", ylabels, [ORANGE, BLUE], ["o", "s"], reference=ref, xlabel=title, title=title)
        add_panel_label(ax, panel)
        if panel in {"B", "C"}:
            ax.tick_params(axis="y", labelleft=False)
    fig.text(0.99, 0.025, "Rate-verify-only without confirmed action: non-estimable (n=0). Descriptive recorded treatment only; no causal interpretation.", ha="right", fontsize=6.1, color=GREY)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.90, bottom=0.20, wspace=0.55)
    save_figure(fig, "FigS7_early_vasopressor_descriptive")


def supplement_operation_selection() -> None:
    perf_path = R1 / "first_vs_all_operations_patient_cluster_sensitivity.csv"
    boot_path = R1 / "first_vs_all_operations_patient_bootstrap_2000.csv"
    perf = read_csv(perf_path)
    boot = read_csv(boot_path)
    states = ["first_fully_evaluable_operation_primary_rule", "all_fully_evaluable_stage2_operations"]
    labels = {states[0]: "First operation (primary)", states[1]: "All operations (patient-clustered)"}
    metrics = ["auroc", "auprc", "brier", "calibration_slope"]
    source = boot.loc[(boot["kind"] == "model") & boot["operation_selection_rule"].isin(states) & boot["metric"].isin(metrics)].copy()
    source["display_label"] = source["operation_selection_rule"].map(labels)
    source = source.merge(perf[["operation_selection_rule", "operations", "patients", "patients_with_multiple_included_operations", "estimability_status"]], on="operation_selection_rule", how="left", validate="many_to_one")
    write_source("FigS8", "A-D", "FigS8_first_vs_all_operations", source, [perf_path, boot_path], "secondary operation-selection sensitivity", "All-operation uncertainty sampled at patient level; primary analysis uses the first fully evaluable operation.")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.5))
    spec = [("auroc", "A", "AUROC", None), ("auprc", "B", "AUPRC", None), ("brier", "C", "Brier score", None), ("calibration_slope", "D", "Calibration slope", 1.0)]
    for ax, (metric, panel, title, ref) in zip(axs.flat, spec):
        d = source.loc[source["metric"] == metric].set_index("operation_selection_rule").loc[states].reset_index()
        ax.errorbar([0, 1], d["estimate"], yerr=[d["estimate"] - d["ci_lower"], d["ci_upper"] - d["estimate"]], fmt="o-", color=BLUE, ecolor=BLUE, capsize=2.2, markersize=4.5)
        if ref is not None:
            ax.axhline(ref, color=GREY, linestyle="--", linewidth=0.8)
        ax.set_xticks([0, 1], [labels[s] for s in states], rotation=12, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title, loc="left", fontweight="bold")
        add_panel_label(ax, panel)
        clean_axis(ax, "y")
    fig.text(0.99, 0.005, "All-operations analysis: 8,491 operations among 7,358 patients; patient-clustered bootstrap 95% CIs.", ha="right", fontsize=6.2, color=GREY)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.19, wspace=0.28, hspace=0.40)
    save_figure(fig, "FigS8_first_vs_all_operations")


def write_captions() -> None:
    text = """# Figure caption drafts

**Figure 1. MOVER external-validation cohort and landmark design.** (A) Flow from the common adult general-anaesthesia cohort to the first fully evaluable operation per patient, followed by the prespecified Stage 1 direct-alert and Stage 2 frozen-model branches. (B) Relative timing of the first H5 hypotension index (t0), the 5-min early-trajectory window, the t0+5 landmark, and the t0+5 to t0+30 primary outcome window. H5 used source observations only, without last-observation-carried-forward or interpolation.

**Figure 2. Primary external validation of the frozen LM5 common-18 model in MOVER.** (A) Restricted-cubic-spline calibration curve with 95% patient-bootstrap confidence band and equal-frequency observed-risk bins. (B) Precision–recall curves; legend values are AUPRC. (C) Decision curves across risk thresholds 0.05–0.50. (D) Alert burden versus event sensitivity for the prespecified two-stage strategy. C15* denotes the primary clinical threshold of 0.15. Error bars are 95% patient-bootstrap confidence intervals.

**Figure 3. H5/R1 observation-process factorial sensitivity.** Four complete-intersection cells cross the feature observation process (H5 or R1) with the outcome observation process (H5 or R1). Panels A and B show cell AUROC and AUPRC. Panels C and D show marginal feature-process, marginal outcome-process, and feature-by-outcome interaction effects. Negative differences in AUROC or AUPRC favor H5 over R1. Error bars are 95% paired patient-bootstrap confidence intervals.

**Figure S1. Phase, buffer, operator, and R1 re-detection sensitivities.** AUROC and AUPRC under full cohort re-selection for each observation variant. H5 phase 0 is the primary reference.

**Figure S2. Endpoint and outcome-coverage sensitivities.** Performance under the primary endpoint and four prespecified endpoint or coverage variants. Error bars are 95% patient-bootstrap confidence intervals.

**Figure S3. Prespecified subgroup performance.** AUROC and calibration slope across all retained prespecified subgroups. Event prevalence is shown in the labels. Error bars are 95% patient-bootstrap confidence intervals.

**Figure S4. Feature drift from INSPIRE to MOVER.** Directional standardized mean differences and MOVER missingness or values outside the INSPIRE 1st–99th percentile range. NE denotes non-estimable standardized mean difference because of zero pooled variance.

**Figure S5. Independent U1/U2 local-update evaluation.** Intercept-only and intercept-plus-slope updates were fitted in one patient half and evaluated in the independent other half. Error bars are 95% paired patient-bootstrap confidence intervals.

**Figure S6. Outcome-observability inverse-probability-weighting sensitivity.** Complete-outcome and P1–P99 truncated IPW point estimates. The prespecified >20% missing-outcome gate was not triggered. Confidence intervals were not available in the locked result layer.

**Figure S7. Early vasopressor action strata (descriptive only).** Performance among operations with a confirmed new bolus or rate change and among operations with no candidate vasopressor action recorded. This analysis is descriptive and does not support causal interpretation.

**Figure S8. First-operation versus all-operation sensitivity.** Primary first-operation estimates are compared with all eligible operations using patient-clustered bootstrap confidence intervals.
"""
    (FIG_DIR / "figure_caption_drafts.md").write_text(text, encoding="utf-8")


def finalize_manifest_and_qa() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    script_hash = sha256_file(Path(__file__))
    for row in MANIFEST_ROWS:
        fig_id = str(row["figure_id"])
        if fig_id not in FIGURE_OUTPUTS:
            raise RuntimeError(f"No rendered outputs registered for {fig_id}")
        png, pdf = FIGURE_OUTPUTS[fig_id]
        row.update(
            {
                "figure_png": relative(png),
                "figure_png_sha256": sha256_file(png),
                "figure_pdf": relative(pdf),
                "figure_pdf_sha256": sha256_file(pdf),
                "generation_script": relative(Path(__file__)),
                "generation_script_sha256": script_hash,
                "generated_at_utc": generated,
            }
        )
    manifest = pd.DataFrame(MANIFEST_ROWS)
    manifest_path = FIG_DIR / "figure_source_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    qa_rows = []
    for fig_id, (png, pdf) in FIGURE_OUTPUTS.items():
        with Image.open(png) as im:
            dpi = im.info.get("dpi", (0, 0))
            width, height = im.size
            png_ok = width >= 1500 and height >= 900 and min(dpi) >= 299
        pdf_ok = pdf.stat().st_size > 1000 and pdf.read_bytes()[:4] == b"%PDF"
        qa_rows.append(
            {
                "figure_id": fig_id,
                "png_width_px": width,
                "png_height_px": height,
                "png_dpi_x": float(dpi[0]),
                "png_dpi_y": float(dpi[1]),
                "png_300dpi_and_size_check": bool(png_ok),
                "pdf_size_bytes": pdf.stat().st_size,
                "pdf_header_check": bool(pdf_ok),
            }
        )
    source_files = sorted(SOURCE_DIR.glob("*.csv"))
    source_checks = []
    for path in source_files:
        d = pd.read_csv(path)
        source_checks.append(
            {
                "file": relative(path),
                "rows": len(d),
                "columns": len(d.columns),
                "contains_patient_or_case_hash": any(c in {"patient_hash", "case_hash"} for c in d.columns),
                "sha256": sha256_file(path),
            }
        )
    if any(x["contains_patient_or_case_hash"] for x in source_checks):
        raise RuntimeError("Figure source data contains patient/case hashes")
    if not all(x["png_300dpi_and_size_check"] and x["pdf_header_check"] for x in qa_rows):
        raise RuntimeError("Rendered figure QA failed")
    qa = {
        "generated_at_utc": generated,
        "u0_hash_manifest_verified_before_read": True,
        "model_refit_or_prediction_generation_performed": False,
        "patient_level_figure_source_exported": False,
        "figure_count": len(qa_rows),
        "source_data_file_count": len(source_checks),
        "figures": qa_rows,
        "source_data": source_checks,
        "manifest_sha256": sha256_file(manifest_path),
        "script_sha256": script_hash,
        "status": "PASS",
    }
    qa_path = FIG_DIR / "figure_QA_report.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "# Figure QA report",
        "",
        "Status: **PASS**",
        "",
        "- U0 result hashes verified before reading any locked U0 output.",
        "- No model fitting, recalibration, or new patient prediction was performed.",
        "- No patient_hash or case_hash was exported to figure source data.",
        f"- {len(qa_rows)} figures each have a 300 dpi PNG and vector PDF.",
        f"- {len(source_checks)} panel-level source-data CSV files are listed in `figure_source_manifest.csv`.",
        "- Automated checks covered dimensions, DPI metadata, PDF headers, nonempty source data, and source-data privacy.",
        "- Visual inspection is recorded separately after rendering.",
    ]
    (FIG_DIR / "figure_QA_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    verify_u0_manifest()
    figure1()
    figure2()
    figure3()
    supplement_phase_operator()
    supplement_endpoints()
    supplement_subgroups()
    supplement_drift()
    supplement_updates()
    supplement_ipw()
    supplement_vasopressor()
    supplement_operation_selection()
    write_captions()
    finalize_manifest_and_qa()
    print(json.dumps({"status": "PASS", "figures": sorted(FIGURE_OUTPUTS), "source_data_files": len(list(SOURCE_DIR.glob('*.csv')))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
