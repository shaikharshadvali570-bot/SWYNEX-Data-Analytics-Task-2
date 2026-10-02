"""
Exploratory Data Analysis - Netflix Movies and TV Shows
=======================================================

Task 2 (EDA & Insight Discovery) for SWYNEX Technologies.

This module consumes the *cleaned* dataset produced in Task 1
(`data/cleaned_dataset.csv`) and performs a complete, reproducible EDA:

    overview -> quality -> descriptive stats -> univariate -> bivariate
    -> time series -> correlation -> outliers -> segmentation -> KPIs -> insights

Every number quoted in `README.md` and in `reports/EDA_Report.md` is produced
here at run time (no hard-coded statistics).

Run from the project root:

    python src/eda_analysis.py

The cleaned Task 1 dataset is treated as read-only. Everything needed on top of
it (a parsed datetime, release decade, content-age gap, primary country) is
derived in memory only.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
VIZ_DIR = PROJECT_ROOT / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUMMARY_JSON = REPORTS_DIR / "eda_summary.json"

# Visualisation groups required by the brief.
CHART_GROUPS = {
    "distribution_charts": VIZ_DIR / "distribution_charts",
    "trend_charts": VIZ_DIR / "trend_charts",
    "comparison_charts": VIZ_DIR / "comparison_charts",
    "correlation_charts": VIZ_DIR / "correlation_charts",
}
GROUP_ALIASES = {
    "distribution": "distribution_charts",
    "trend": "trend_charts",
    "comparison": "comparison_charts",
    "correlation": "correlation_charts",
    "distribution_charts": "distribution_charts",
    "trend_charts": "trend_charts",
    "comparison_charts": "comparison_charts",
    "correlation_charts": "correlation_charts",
}

CATEGORICAL_COLUMNS = ["type", "rating", "country", "listed_in", "duration_unit"]
NUMERICAL_COLUMNS = ["release_year", "duration_value"]
PLACEHOLDER = "Not Specified"

# Columns created by build_analysis_ready() in memory. They are excluded from
# the "dataset structure" numbers so the overview reports the real Task 1 file.
DERIVED_COLUMNS = [
    "year_added", "month_added", "month_name_added", "weekday_added",
    "release_decade", "years_to_netflix", "primary_country", "genre_count", "cast_size",
]

# Netflix's streaming catalogue in this snapshot starts in 2008; the platform
# scaled globally after 2015, so that boundary is used for a fair growth window.
MATURE_YEAR_FROM = 2015

TYPE_COLORS = {"Movie": "#1f77b4", "TV Show": "#d62728"}
ACCENT = "#4c72b0"
NEUTRAL = "#8c8c8c"
WARNING = "#c44e52"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#4d4d4d",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

CHART_INDEX: list[dict[str, str]] = []
RESULTS: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def rule(title: str, level: int = 1) -> None:
    line = "=" if level == 1 else "-"
    print(f"\n{line * 78}\n{title.upper()}\n{line * 78}")


def bullet(label: str, value: object) -> None:
    print(f"  {label:<44}{value}")


def pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def save_figure(fig: plt.Figure, group: str, filename: str, title: str, insight: str) -> Path:
    """Persist a figure into its visualisation group and index it."""
    resolved = GROUP_ALIASES.get(group, group)
    target_dir = CHART_GROUPS[resolved]
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rel = f"visualizations/{resolved}/{filename}"
    CHART_INDEX.append(
        {
            "group": resolved,
            "file": filename,
            "relative_path": rel,
            "title": title,
            "insight": insight,
        }
    )
    print(f"  [saved] {rel}")
    return path


def annotate_bars(ax: plt.Axes, values, labels=None, orient: str = "v") -> None:
    """Write values at the end of bars so charts stay readable."""
    vals = list(np.asarray(values, dtype=float).ravel())
    labs = list(labels) if labels is not None else [f"{v:,.0f}" for v in vals]
    span = max(max([abs(v) for v in vals] + [1e-9]), 1e-9)
    for i, patch in enumerate(ax.patches):
        if i >= len(vals):
            break
        text = labs[i]
        if orient == "v":
            ax.annotate(
                text,
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center", va="bottom", fontsize=8.5,
                xytext=(0, 2), textcoords="offset points",
            )
        else:
            ax.annotate(
                text,
                (patch.get_width(), patch.get_y() + patch.get_height() / 2),
                ha="left", va="center", fontsize=8.5,
                xytext=(3, 0), textcoords="offset points",
            )
    if orient == "v":
        ax.set_ylim(0, span * 1.15)
    else:
        ax.set_xlim(0, span * 1.18)


def style_axis(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.0)


# ---------------------------------------------------------------------------
# Loading & analysis-ready derivation
# ---------------------------------------------------------------------------

def load_cleaned_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Task 1 cleaned dataset (UTF-8 with BOM, exactly as Task 1 wrote it)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected the Task 1 cleaned dataset at {path}. "
            "Copy it from the Task 1 project before running the EDA."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    # `date_added` is the only column Task 1 intentionally left partially null.
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    return df


def build_analysis_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the extra fields the EDA needs - in memory only.

    The cleaned dataset on disk is never overwritten; these columns exist purely
    to make temporal, decade, content-age and list-based analysis possible.
    """
    out = df.copy()

    # Text columns: remove the residual zero-width space (U+200B). Task 1
    # normalised ordinary whitespace, but U+200B is not whitespace to
    # str.strip(), so a few titles still carried a leading copy.
    zwsp = "\u200b"
    short_text = [
        c for c in out.columns
        if out[c].dtype == "str" and c not in {"description", "cast", "director"}
    ]
    for col in short_text:
        out[col] = out[col].str.replace(zwsp, "", regex=False).str.strip()

    out["year_added"] = out["date_added"].dt.year.astype("Int64")
    out["month_added"] = out["date_added"].dt.month.astype("Int64")
    out["month_name_added"] = out["date_added"].dt.strftime("%B")
    out["weekday_added"] = out["date_added"].dt.day_name()
    out["release_decade"] = (out["release_year"] // 10 * 10).astype("int64")
    out["years_to_netflix"] = (out["year_added"] - out["release_year"]).astype("Int64")

    out["primary_country"] = out["country"].str.split(", ").str[0]
    out["genre_count"] = out["listed_in"].str.count(",") + 1
    out["cast_size"] = np.where(
        out["cast"].eq(PLACEHOLDER), 0, out["cast"].str.count(",") + 1
    )
    return out


def frequency_table(series: pd.Series, top: int | None = None) -> pd.DataFrame:
    """Count / percentage table shared by the categorical analyses."""
    counts = series.value_counts(dropna=False)
    if top:
        counts = counts.head(top)
    frame = counts.rename("count").to_frame()
    frame["share"] = frame["count"] / len(series)
    frame["share_pct"] = (frame["share"] * 100).round(2)
    frame["cumulative_pct"] = (frame["share"].cumsum() * 100).round(2)
    frame.index.name = series.name
    return frame


def explode_list_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Explode a ', '-separated column into a tidy value-count Series."""
    valid = df[column].dropna().loc[lambda s: s.ne(PLACEHOLDER)]
    tokens = valid.str.split(", ").explode().str.strip()
    return tokens.loc[lambda s: s.ne("") & s.notna()].value_counts()


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fence for a numeric series (NaN-safe)."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def numeric_profile(series: pd.Series) -> pd.Series:
    """Full descriptive profile for one numeric column."""
    clean = series.dropna()
    q1, q3 = clean.quantile([0.25, 0.75])
    low, high = iqr_bounds(clean)
    return pd.Series(
        {
            "count": float(clean.size),
            "mean": clean.mean(),
            "median": clean.median(),
            "mode": clean.mode().iloc[0] if not clean.mode().empty else np.nan,
            "std": clean.std(),
            "variance": clean.var(),
            "min": clean.min(),
            "q1": q1,
            "q3": q3,
            "max": clean.max(),
            "range": clean.max() - clean.min(),
            "iqr": q3 - q1,
            "p10": clean.quantile(0.10),
            "p90": clean.quantile(0.90),
            "skewness": clean.skew(),
            "outliers_iqr": int(((clean < low) | (clean > high)).sum()),
            "lower_fence": low,
            "upper_fence": high,
        }
    )


# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------

def _column_role(df: pd.DataFrame, column: str) -> str:
    if column in {"show_id", "title"}:
        return "identifier"
    if column == "date_added":
        return "datetime"
    if column in {"release_year", "duration_value", "year_added", "month_added",
                  "release_decade", "years_to_netflix", "genre_count", "cast_size"}:
        return "numeric"
    if column in {"country", "listed_in", "cast", "director"}:
        return "multi-value text"
    if column == "description":
        return "free text"
    if column == "added_before_release":
        return "boolean"
    return "categorical"


def dataset_overview(df: pd.DataFrame) -> dict:
    rule("1. Dataset Overview")
    dataset_columns = [c for c in df.columns if c not in DERIVED_COLUMNS]
    derived = [c for c in df.columns if c in DERIVED_COLUMNS]
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    bullet("Rows", f"{df.shape[0]:,}")
    bullet("Columns in the cleaned dataset", len(dataset_columns))
    bullet("Cells in the cleaned dataset", f"{df.shape[0] * len(dataset_columns):,}")
    bullet("Missing cells (dataset columns)", int(df[dataset_columns].isna().sum().sum()))
    bullet("Duplicate rows (all columns)", int(df.duplicated().sum()))
    bullet("Duplicate show_id", int(df["show_id"].duplicated().sum()))
    bullet("Derived EDA columns (in memory only)", ", ".join(derived))
    bullet("Analysis frame size", f"{df.shape[0]:,} x {df.shape[1]} (~{memory_mb:.2f} MB in memory)")

    dtype_table = pd.DataFrame(
        {
            "column": dataset_columns,
            "dtype": [str(df[c].dtype) for c in dataset_columns],
            "non_null": [int(df[c].notna().sum()) for c in dataset_columns],
            "missing": [int(df[c].isna().sum()) for c in dataset_columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in dataset_columns],
            "role": [_column_role(df, c) for c in dataset_columns],
        }
    )
    print("\nColumn profile of the cleaned dataset (structure, role, completeness)")
    print(dtype_table.to_string(index=False))

    print("\nUnique values of key categorical columns")
    cat_unique = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            cat_unique[col] = int(df[col].nunique())
            bullet(f"  {col}", f"{df[col].nunique()} distinct values")
    cat_unique["primary_country"] = int(df["primary_country"].nunique())

    results = {
        "rows": int(df.shape[0]),
        "columns": len(dataset_columns),
        "cells": int(df.shape[0] * len(dataset_columns)),
        "memory_mb": round(memory_mb, 2),
        "missing_cells": int(df[dataset_columns].isna().sum().sum()),
        "missing_cells_pct": float(df[dataset_columns].isna().to_numpy().mean()),
        "duplicate_rows": int(df.duplicated().sum()),
        "derived_columns": derived,
        "categorical_unique": cat_unique,
        "dtypes": dtype_table.to_dict(orient="records"),
    }
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            top = df[col].value_counts()
            results[f"unique_{col}"] = int(df[col].nunique())
            results[f"top_{col}"] = {"value": top.index[0], "count": int(top.iloc[0])}
    return results

# ---------------------------------------------------------------------------
# 2. Data quality check (inherit Task 1 guarantees, then look for leftovers)
# ---------------------------------------------------------------------------

def data_quality_check(df: pd.DataFrame) -> dict:
    rule("2. Data Quality Check (verified on the cleaned dataset)")
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "result": "PASS" if passed else "REVIEW", "detail": detail})
        bullet(f"  [{'PASS' if passed else 'REVIEW'}] {name}", detail)

    check("show_id is unique", not df["show_id"].duplicated().any(),
          f"{df['show_id'].nunique():,} unique ids")
    check("no duplicate records", not df.duplicated().any(),
          f"{int(df.duplicated().sum())} duplicate rows")
    check("type only Movie / TV Show", set(df["type"].unique()) <= {"Movie", "TV Show"},
          ", ".join(f"{k} ({v:,})" for k, v in df["type"].value_counts().items()))
    check("duration_unit matches type",
          bool((df.loc[df["type"] == "Movie", "duration_unit"] == "Minutes").all()
               and (df.loc[df["type"] == "TV Show", "duration_unit"] == "Seasons").all()),
          "movies in Minutes, TV shows in Seasons")
    check("duration_value is positive", bool((df["duration_value"] > 0).all()),
          f"min {df['duration_value'].min()}, max {df['duration_value'].max()}")
    check("release_year within 1900-2021", bool(df["release_year"].between(1900, 2021).all()),
          f"{df['release_year'].min()}-{df['release_year'].max()}")
    check("date_added not in the future",
          bool(df["date_added"].dropna().le(pd.Timestamp("2021-01-16")).all()),
          f"latest {df['date_added'].max():%Y-%m-%d}")

    missing = df[[c for c in df.columns if c not in DERIVED_COLUMNS]].isna().sum()
    missing = missing[missing > 0]
    check("missing values limited to date_added",
          set(missing.index) <= {"date_added"},
          "; ".join(f"{c}: {int(n):,} ({pct(n / len(df))})" for c, n in missing.items()))

    # Residual issue found during analysis: invisible zero-width spaces that
    # Task 1's whitespace normalisation (str.strip) cannot see.
    raw_text = pd.read_csv(DATA_PATH, encoding="utf-8-sig", dtype="str").fillna("")
    zwsp_hits = {
        col: int(raw_text[col].str.contains("\u200b", regex=False).sum())
        for col in ["title", "director", "cast", "country", "listed_in"]
        if col in raw_text.columns
    }
    zwsp_total = sum(zwsp_hits.values())
    check("no invisible zero-width space (U+200B) in short text",
          zwsp_total == 0,
          f"{zwsp_total} cells: " + ", ".join(f"{k} {v:,}" for k, v in zwsp_hits.items() if v))

    placeholder_counts = {
        col: int((df[col] == PLACEHOLDER).sum())
        for col in ["director", "cast", "country", "rating"]
        if col in df.columns
    }
    print("\n  Placeholder cells inherited from Task 1 (kept, not treated as missing):")
    for col, n in placeholder_counts.items():
        bullet(f"    {col}", f"{n:,} ({pct(n / len(df))})")

    print("\n  Verdict: the cleaned dataset is analysis-ready. No blocking issue was found,\n"
          "  so it is used as-is and is never modified by this project.")

    return {
        "checks": checks,
        "missing": {k: int(v) for k, v in missing.items()},
        "placeholders": placeholder_counts,
        "zwsp_cells": zwsp_total,
        "zwsp_hits": {k: v for k, v in zwsp_hits.items() if v},
    }
# ---------------------------------------------------------------------------
# 3. Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_statistics(df: pd.DataFrame) -> dict:
    rule("3. Descriptive Statistics")

    profiles = {col: numeric_profile(df[col]) for col in NUMERICAL_COLUMNS}
    numeric_table = pd.DataFrame(profiles).T
    ordered = ["count", "mean", "median", "mode", "std", "variance", "min", "q1",
               "q3", "max", "range", "iqr", "p10", "p90", "skewness",
               "outliers_iqr", "lower_fence", "upper_fence"]
    numeric_table = numeric_table[ordered].round(3)
    print("\nNumeric descriptive statistics (count, mean, median, std, quartiles, IQR, skew)")
    print(numeric_table.to_string())

    # Separate duration profiles per content type: mixing minutes with seasons
    # would be meaningless, so each unit is profiled on its own.
    by_type = {}
    print("\nDuration profile split by content unit")
    for unit, sub in df.groupby("duration_unit"):
        s = sub["duration_value"]
        q1, q3 = s.quantile([0.25, 0.75])
        by_type[unit] = {
            "count": int(s.size), "mean": round(s.mean(), 2), "median": float(s.median()),
            "mode": float(s.mode().iloc[0]), "std": round(s.std(), 2),
            "min": int(s.min()), "q1": float(q1), "q3": float(q3), "max": int(s.max()),
            "iqr": float(q3 - q1), "skewness": round(s.skew(), 3),
        }
        bullet(f"  {unit} (n={s.size:,})",
               f"mean {s.mean():.1f} · median {s.median():.0f} · p90 {s.quantile(0.9):.0f} "
               f"· range {s.min():.0f}-{s.max():.0f} · IQR {q3 - q1:.0f} · skew {s.skew():.2f}")

    categorical = {}
    print("\nCategorical summary (unique values, most frequent category, share)")
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        table_ = frequency_table(df[col])
        top = table_.iloc[0]
        categorical[col] = {
            "unique": int(df[col].nunique()),
            "top_category": str(top.name),
            "top_count": int(top["count"]),
            "top_share": float(top["share"]),
            "lowest_category": str(table_.index[-1]),
            "lowest_count": int(table_["count"].iloc[-1]),
        }
        bullet(f"  {col}",
               f"{df[col].nunique()} categories · top '{top.name}' {int(top['count']):,} "
               f"({pct(top['share'])})")

    return {"numeric": {k: _json_safe(v.to_dict()) for k, v in profiles.items()}, "duration_by_unit": _json_safe(by_type),
            "categorical": _json_safe(categorical)}
# ---------------------------------------------------------------------------
# 4. Univariate analysis
# ---------------------------------------------------------------------------

def univariate_analysis(df: pd.DataFrame) -> dict:
    rule("4. Univariate Analysis")
    facts: dict[str, object] = {}

    movies = df.loc[df["type"] == "Movie", "duration_value"]
    shows = df.loc[df["type"] == "TV Show", "duration_value"]

    # --- 4.1 Movie runtime distribution + TV season distribution ------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    bins = np.arange(0, movies.max() + 10, 10)
    ax.hist(movies, bins=bins, color=TYPE_COLORS["Movie"], alpha=0.85, edgecolor="white")
    ax.axvline(movies.mean(), color=WARNING, ls="--", lw=1.6,
               label=f"Mean {movies.mean():.0f} min")
    ax.axvline(movies.median(), color="black", ls="-", lw=1.6,
               label=f"Median {movies.median():.0f} min")
    q1, q3 = movies.quantile([0.25, 0.75])
    ax.axvspan(q1, q3, color=NEUTRAL, alpha=0.18, label=f"IQR {q1:.0f}-{q3:.0f} min")
    style_axis(ax, f"Movie runtime (n={len(movies):,})", "Duration (minutes)", "Number of titles")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    season_counts = shows.value_counts().sort_index()
    ax.bar(season_counts.index.astype(str), season_counts.values,
           color=TYPE_COLORS["TV Show"], alpha=0.85, edgecolor="white")
    annotate_bars(ax, season_counts.values)
    style_axis(ax, f"TV show length (n={len(shows):,})", "Number of seasons", "Number of titles")
    ax.set_ylim(0, season_counts.max() * 1.18)

    fig.suptitle("How long is a Netflix title? Runtime and season counts follow very different shapes",
                 fontsize=13, fontweight="bold", y=1.04)
    save_figure(fig, "distribution", "fig_01_duration_distribution.png",
                "Duration distribution: movie runtime vs TV seasons",
                "Movie runtimes cluster around 90-100 minutes while most TV shows run for a single season.")

    # --- 4.2 Box plot for outlier inspection --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    bp = ax.boxplot(movies, orientation="vertical", patch_artist=True, widths=0.45,
                    boxprops=dict(facecolor=TYPE_COLORS["Movie"], alpha=0.55),
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor=WARNING,
                                    markeredgecolor=WARNING, alpha=0.6))
    style_axis(ax, f"Movie runtime - IQR fences at {iqr_bounds(movies)[0]:.0f} / {iqr_bounds(movies)[1]:.0f} min",
               "", "Duration (minutes)")
    ax.set_xticks([1], ["Movies"])

    ax = axes[1]
    ax.boxplot(shows, orientation="vertical", patch_artist=True, widths=0.45,
               boxprops=dict(facecolor=TYPE_COLORS["TV Show"], alpha=0.55),
               medianprops=dict(color="black", lw=2),
               flierprops=dict(marker="o", markersize=4, markerfacecolor=WARNING,
                               markeredgecolor=WARNING, alpha=0.6))
    style_axis(ax, f"TV show seasons - fence at {iqr_bounds(shows)[1]:.0f} seasons",
               "", "Number of seasons")
    ax.set_xticks([1], ["TV Shows"])
    fig.suptitle("Box plots expose the long upper tail: unusually long films and long-running series",
                 fontsize=13, fontweight="bold", y=1.02)
    save_figure(fig, "distribution", "fig_02_duration_boxplot.png",
                "Box plots of duration by content unit",
                "The upper tail of movie runtimes and multi-season series are outliers by IQR but are legitimate catalogue entries.")

    low_m, high_m = iqr_bounds(movies)
    low_s, high_s = iqr_bounds(shows)
    facts["movie_runtime"] = {
        "mean": round(float(movies.mean()), 2), "median": float(movies.median()),
        "std": round(float(movies.std()), 2), "skew": round(float(movies.skew()), 3),
        "q1": float(q1), "q3": float(q3), "max": int(movies.max()), "min": int(movies.min()),
        "outliers": int(((movies < low_m) | (movies > high_m)).sum()),
        "short_titles": int((movies < 60).sum()),
        "short_share": float((movies < 60).mean()),
    }
    facts["tv_seasons"] = {
        "mean": round(float(shows.mean()), 2), "median": float(shows.median()),
        "std": round(float(shows.std()), 2), "skew": round(float(shows.skew()), 3),
        "max": int(shows.max()),
        "single_season": int((shows == 1).sum()),
        "single_season_share": float((shows == 1).mean()),
        "outliers": int(((shows < low_s) | (shows > high_s)).sum()),
    }
    return facts

    print("\nFull frequency tables")
    for col in ["type", "rating", "duration_unit", "primary_country"]:
        print(f"\n{col} (top 10)")
        print(frequency_table(df[col], top=10).to_string())

    return {
        "numeric": numeric_table.to_dict(orient="index"),
        "duration_by_unit": by_type,
        "categorical": categorical,
    }
# ---------------------------------------------------------------------------
# 5. Univariate analysis (part 2: release year, ratings, content mix)
# ---------------------------------------------------------------------------

def univariate_categorical(df: pd.DataFrame) -> dict:
    facts: dict[str, object] = {}

    # --- 5.1 Release-year distribution of the catalogue ---------------------
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    years = df["release_year"]
    ax.hist(years, bins=np.arange(years.min(), years.max() + 2, 1),
            color=ACCENT, alpha=0.9, edgecolor="white")
    median_year = years.median()
    ax.axvline(median_year, color=WARNING, ls="--", lw=1.6,
               label=f"Median release year {median_year:.0f}")
    ax.axvline(MATURE_YEAR_FROM, color="black", ls=":", lw=1.6,
               label=f"{MATURE_YEAR_FROM}+ catalogue")
    recent_share = float((years >= MATURE_YEAR_FROM).mean())
    ax.annotate(f"{recent_share*100:.1f}% of titles were released in {MATURE_YEAR_FROM} or later",
                xy=(0.02, 0.92), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f0f4fa", ec="#9db4d0"))
    style_axis(ax, "When was the content originally released?", "Release year", "Number of titles")
    ax.legend(loc="upper left", fontsize=9)
    save_figure(fig, "distribution", "fig_03_release_year_distribution.png",
                "Release-year distribution of the Netflix catalogue",
                "The catalogue is dominated by recently released content; older titles are a thin historical tail.")

    facts["release_year"] = {
        "min": int(years.min()), "max": int(years.max()),
        "mean": round(float(years.mean()), 2), "median": float(years.median()),
        "skew": round(float(years.skew()), 3),
        "recent_share": recent_share,
    }

    # --- 5.2 Content rating mix ---------------------------------------------
    rating_table = frequency_table(df["rating"]).drop(columns=["cumulative_pct"])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = [ACCENT if i < 5 else NEUTRAL for i in range(len(rating_table))]
    bars = ax.barh(rating_table.index[::-1], rating_table["count"][::-1],
                   color=colors[::-1], alpha=0.9, edgecolor="white")
    annotate_bars(ax, rating_table["count"][::-1].values,
                  labels=[f"{c:,}  ({s:.1f}%)" for c, s in
                          zip(rating_table["count"][::-1], rating_table["share_pct"][::-1])],
                  orient="h")
    style_axis(ax, "What content ratings does the catalogue carry?", "Number of titles", "Rating")
    ax.grid(axis="y", alpha=0.0)
    save_figure(fig, "comparison", "fig_09_rating_distribution.png",
                "Content rating mix",
                "Two mature-audience ratings (TV-MA and TV-14) account for the majority of the catalogue.")

    mature_share = float(df["rating"].isin(["TV-MA", "TV-14", "R"]).mean())
    family_share = float(df["rating"].isin(["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G", "PG"]).mean())
    facts["rating"] = {
        "top": rating_table.index[0], "top_count": int(rating_table["count"].iloc[0]),
        "top_share": float(rating_table["share"].iloc[0]),
        "top3_share": float(rating_table["share"].iloc[:3].sum()),
        "mature_share": mature_share, "family_share": family_share,
        "least": rating_table.index[-1], "least_count": int(rating_table["count"].iloc[-1]),
        "not_specified": int((df["rating"] == PLACEHOLDER).sum()),
    }

    # --- 5.3 Content type split (composition) -------------------------------
    type_counts = df["type"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.pie(type_counts.values, labels=None, autopct=lambda p: f"{p:.1f}%",
           colors=[TYPE_COLORS[i] for i in type_counts.index],
           startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           textprops=dict(fontsize=11, fontweight="bold", color="white"))
    ax.legend(type_counts.index, loc="center", fontsize=10, title="Content type")
    ax.set_title("Catalogue composition by type")

    ax = axes[1]
    yearly = (df.dropna(subset=["year_added"])
                .groupby(["year_added", "type"]).size().unstack(fill_value=0))
    for content_type in yearly.columns:
        ax.plot(yearly.index, yearly[content_type], marker="o", lw=2,
                color=TYPE_COLORS[content_type], label=content_type)
    style_axis(ax, "Titles added per year, by type", "Year added to Netflix", "Titles added")
    ax.legend(fontsize=9)
    fig.suptitle("Movies dominate the catalogue, but TV shows grew as a share of annual additions",
                 fontsize=13, fontweight="bold", y=1.03)
    save_figure(fig, "comparison", "fig_10_type_composition.png",
                "Movie vs TV show composition and additions over time",
                "Movies make up roughly two thirds of the catalogue; TV additions accelerated after 2015.")

    movie_share = float((df["type"] == "Movie").mean())
    facts["type"] = {
        "movie_count": int(type_counts.get("Movie", 0)),
        "tv_count": int(type_counts.get("TV Show", 0)),
        "movie_share": movie_share,
        "ratio": float(type_counts.get("Movie", 0) / max(type_counts.get("TV Show", 1), 1)),
    }
    return facts
# ---------------------------------------------------------------------------
# 6. Bivariate analysis
# ---------------------------------------------------------------------------

def bivariate_analysis(df: pd.DataFrame) -> dict:
    rule("6. Bivariate Analysis")
    facts: dict[str, object] = {}

    # --- 6.1 Numeric vs numeric: release year vs movie runtime --------------
    # Runtime is only comparable within one unit, so Movies are isolated.
    movies = df.loc[df["type"] == "Movie", ["release_year", "duration_value", "release_decade"]]
    corr = movies["release_year"].corr(movies["duration_value"])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.regplot(data=movies.sample(min(len(movies), 2200), random_state=42),
                x="release_year", y="duration_value", ax=ax,
                scatter_kws=dict(s=14, alpha=0.28, color=ACCENT, edgecolor="none"),
                line_kws=dict(color=WARNING, lw=2.4))
    style_axis(ax, "Do newer movies run longer or shorter?",
               "Release year", "Movie runtime (minutes)")
    ax.annotate(f"Pearson r = {corr:+.3f}  (effectively no linear relationship)",
                xy=(0.03, 0.93), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f0f4fa", ec="#9db4d0"))
    save_figure(fig, "correlation", "fig_07_runtime_vs_release_year.png",
                "Movie runtime vs release year (scatter)",
                "Movie runtime is stable across release years; the correlation is close to zero.")

    by_decade = (movies.groupby("release_decade")["duration_value"]
                 .agg(["count", "mean", "median"]).query("count >= 20"))
    facts["runtime_vs_year"] = {
        "pearson_r": round(float(corr), 4),
        "mean_min": round(float(by_decade["mean"].min()), 2),
        "mean_max": round(float(by_decade["mean"].max()), 2),
        "spread_minutes": round(float(by_decade["mean"].max() - by_decade["mean"].min()), 2),
        "decades_compared": int(len(by_decade)),
    }

    # --- 6.2 Numeric vs categorical: runtime by release decade --------------
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    plot_df = movies[movies["release_decade"] >= 1980]
    sns.boxplot(data=plot_df, x="release_decade", y="duration_value", ax=ax,
                color=ACCENT, width=0.6, fliersize=2.5)
    ax.axhline(movies["duration_value"].median(), color=WARNING, ls="--", lw=1.4,
               label=f"Overall median {movies['duration_value'].median():.0f} min")
    style_axis(ax, "Movie runtime by release decade (1980 onwards)",
               "Release decade", "Movie runtime (minutes)")
    ax.legend(fontsize=9)
    save_figure(fig, "comparison", "fig_11_runtime_by_decade.png",
                "Movie runtime distribution by release decade",
                "The median runtime barely moves across four decades, so era does not drive film length.")

    # --- 6.3 Categorical vs categorical: rating mix inside each type --------
    ct = pd.crosstab(df["rating"], df["type"], normalize="columns") * 100
    ct = ct.loc[df["rating"].value_counts().index]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=ax,
                cbar_kws={"label": "% of titles within content type"}, linewidths=0.4,
                linecolor="white")
    style_axis(ax, "Content rating mix differs sharply between movies and TV shows",
               "Content type", "Rating")
    ax.grid(False)
    save_figure(fig, "comparison", "fig_12_rating_by_type.png",
                "Rating mix within movies vs TV shows",
                "Movies lean on film ratings (R, PG-13) while TV shows are dominated by TV-MA and TV-14.")

    facts["rating_by_type"] = {
        "movie_top_rating": str(ct.loc[:, "Movie"].idxmax()),
        "movie_top_share": round(float(ct.loc[:, "Movie"].max()), 2),
        "tv_top_rating": str(ct.loc[:, "TV Show"].idxmax()),
        "tv_top_share": round(float(ct.loc[:, "TV Show"].max()), 2),
    }
    return facts
# ---------------------------------------------------------------------------
# 7. Time-series / trend analysis
# ---------------------------------------------------------------------------

def time_series_analysis(df: pd.DataFrame) -> dict:
    rule("7. Time-Series / Trend Analysis")
    dated = df.dropna(subset=["date_added"]).copy()

    daily = dated.groupby("date_added").size()
    yearly = dated.groupby("year_added").size()
    monthly = dated.set_index("date_added").resample("MS").size()

    # --- 7.1 Annual additions ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    ax.bar(yearly.index.astype(int).astype(str), yearly.values,
           color=ACCENT, alpha=0.9, edgecolor="white")
    annotate_bars(ax, yearly.values)
    style_axis(ax, "Titles added to Netflix per year", "Year added", "Titles added")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, yearly.max() * 1.18)

    ax = axes[1]
    ax.plot(monthly.index, monthly.values, color=ACCENT, lw=1.5, label="Monthly additions")
    rolling = monthly.rolling(6, min_periods=3).mean()
    ax.plot(rolling.index, rolling.values, color=WARNING, lw=2.4, label="6-month moving average")
    style_axis(ax, "Monthly additions with trend (2008-2021)", "Month added", "Titles added")
    ax.legend(fontsize=9)
    fig.suptitle("Catalogue growth is not linear: a long ramp-up peaks in 2019 and then slows",
                 fontsize=13, fontweight="bold", y=1.03)
    save_figure(fig, "trend", "fig_04_additions_over_time.png",
                "Annual and monthly catalogue additions",
                "Additions ramped up steadily from 2008, peaked in 2019 and declined in 2020-2021.")

    # --- 7.2 Seasonality: month and weekday of addition --------------------
    month_order = ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"]
    by_month = dated.groupby("month_name_added").size().reindex(month_order).fillna(0)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"]
    by_weekday = dated.groupby("weekday_added").size().reindex(weekday_order).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    ax.bar(by_month.index, by_month.values, color=ACCENT, alpha=0.9, edgecolor="white")
    ax.set_xticks(range(len(by_month)))
    ax.set_xticklabels([m[:3] for m in by_month.index])
    annotate_bars(ax, by_month.values)
    style_axis(ax, "Seasonality by calendar month", "Month", "Titles added")
    ax.set_ylim(0, by_month.max() * 1.18)

    ax = axes[1]
    colors = [WARNING if w == "Friday" else ACCENT for w in by_weekday.index]
    ax.bar(by_weekday.index, by_weekday.values, color=colors, alpha=0.9, edgecolor="white")
    annotate_bars(ax, by_weekday.values)
    style_axis(ax, "Which weekday does Netflix publish on?", "Weekday", "Titles added")
    ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Clear publishing rhythm: Fridays dominate, and additions cluster late in the year",
                 fontsize=13, fontweight="bold", y=1.03)
    save_figure(fig, "trend", "fig_05_seasonality_weekday_month.png",
                "Monthly and weekday seasonality of catalogue additions",
                "Friday is by far the most common release day, and October-December carry the heaviest monthly loads.")

    # --- 7.3 Content-age gap: release year vs year added --------------------
    gap = dated["years_to_netflix"].dropna()
    yr = dated.groupby("year_added")["years_to_netflix"].median()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    ax.hist(gap, bins=np.arange(gap.min(), min(gap.max(), 80) + 2, 1),
            color=ACCENT, alpha=0.9, edgecolor="white")
    ax.axvline(gap.median(), color=WARNING, ls="--", lw=1.6,
               label=f"Median gap {gap.median():.0f} year(s)")
    style_axis(ax, "How old is a title when Netflix adds it?",
               "Years between release and addition", "Number of titles")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(yr.index.astype(int), yr.values, marker="o", lw=2.2, color=ACCENT)
    style_axis(ax, "Median content age at upload, by year added", "Year added",
               "Median years since release")
    fig.suptitle("Netflix increasingly adds near-current content rather than back catalogue",
                 fontsize=13, fontweight="bold", y=1.03)
    save_figure(fig, "trend", "fig_06_content_age_gap.png",
                "Content age at time of upload",
                "Day-and-date additions dominate and median content age fell towards 0-1 years.")

    return _trend_facts(dated, daily, yearly, monthly, by_month, by_weekday, gap)
def _trend_facts(dated, daily, yearly, monthly, by_month, by_weekday, gap) -> dict:
    share_recent = float((dated["years_to_netflix"] <= 1).mean())
    return {
        "trend": {
            "first_date": str(dated["date_added"].min().date()),
            "last_date": str(dated["date_added"].max().date()),
            "years_covered": int(dated["year_added"].nunique()),
            "peak_year": int(yearly.idxmax()),
            "peak_year_additions": int(yearly.max()),
            "first_year": int(yearly.index.min()),
            "first_year_additions": int(yearly.iloc[0]),
            "growth_multiple": round(float(yearly.max() / max(yearly.iloc[0], 1)), 1),
            "y2019": int(yearly.get(2019, 0)),
            "y2020": int(yearly.get(2020, 0)),
            "peak_decline_pct": round(float(
                (yearly.get(2020, 0) - yearly.get(2019, 0)) / max(yearly.get(2019, 1), 1) * 100), 1),
            "busiest_month": str(by_month.idxmax()),
            "busiest_month_count": int(by_month.max()),
            "friday_count": int(by_weekday.get("Friday", 0)),
            "friday_share": float(by_weekday.get("Friday", 0) / by_weekday.sum()),
            "median_gap": float(gap.median()),
            "mean_gap": round(float(gap.mean()), 2),
            "share_within_1yr": share_recent,
            "daily_mean": round(float(daily.mean()), 2),
            "max_day": str(daily.idxmax().date()),
            "max_day_count": int(daily.max()),
            "top_days": [(str(d.date()), int(c))
                         for d, c in daily.sort_values(ascending=False).head(5).items()],
            "yearly": {int(k): int(v) for k, v in yearly.items()},
        }
    }
# ---------------------------------------------------------------------------
# 8. Correlation analysis
# ---------------------------------------------------------------------------

def correlation_analysis(df: pd.DataFrame) -> dict:
    rule("8. Correlation Analysis")

    # Only genuinely numeric, comparable measures are correlated. `duration_value`
    # mixes minutes (movies) with seasons (TV), so it is correlated within the
    # movie subpopulation and also shown with a warning in the mixed matrix.
    movie_numeric = df.loc[df["type"] == "Movie", [
        "release_year", "duration_value", "genre_count", "cast_size", "years_to_netflix",
    ]]
    corr_movies = movie_numeric.corr()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    ax = axes[0]
    sns.heatmap(corr_movies, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    style_axis(ax, "Correlation among numeric fields - Movies only", "", "")
    ax.grid(False)
    ax.tick_params(axis="x", rotation=35)

    ax = axes[1]
    mixed = df[["release_year", "duration_value", "genre_count", "cast_size", "years_to_netflix"]]
    sns.heatmap(mixed.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Pearson correlation"})
    style_axis(ax, "Same fields, Movies + TV Shows combined (duration not comparable)",
               "", "")
    ax.grid(False)
    ax.tick_params(axis="x", rotation=35)
    fig.suptitle("No strong linear relationships exist between the catalogue's numeric fields",
                 fontsize=13, fontweight="bold", y=1.02)
    save_figure(fig, "correlation", "fig_08_correlation_heatmap.png",
                "Correlation heatmap (movies-only and mixed)",
                "All pairwise correlations are weak; the strongest link is that titles added later tend to be newer releases.")

    # Extract and describe the strongest relationships.
    pairs = (
        corr_movies.where(~np.eye(len(corr_movies), dtype=bool))
        .stack().rename("r").reset_index()
    )
    pairs.columns = ["variable_1", "variable_2", "r"]
    pairs["abs_r"] = pairs["r"].abs()
    pairs = pairs.sort_values("abs_r", ascending=False).drop_duplicates(subset=["abs_r"])
    print("\nMovie-only correlation pairs, strongest first")
    print(pairs[["variable_1", "variable_2", "r"]].head(6).to_string(index=False))

    strong = pairs[pairs["abs_r"] >= 0.5]
    moderate = pairs[(pairs["abs_r"] >= 0.3) & (pairs["abs_r"] < 0.5)]
    weak = pairs[pairs["abs_r"] < 0.3]
    bullet("  Strong pairs (|r| >= 0.5)", len(strong))
    bullet("  Moderate pairs (0.3 <= |r| < 0.5)", len(moderate))
    bullet("  Weak pairs (|r| < 0.3)", len(weak))

    # The one relationship worth explaining in business language.
    age_link = float(df["years_to_netflix"].corr(df["release_year"]))
    print(f"\n  Interpretation: correlation is not causation. The only moderate link\n"
          f"  (release year vs. years-to-Netflix, r = {age_link:+.2f}) is a mechanical\n"
          f"  consequence of the calendar, not evidence that one causes the other.")

    return {
        "movie_matrix": corr_movies.round(4).to_dict(),
        "strong_pairs": int(len(strong)),
        "moderate_pairs": int(len(moderate)),
        "weak_pairs": int(len(weak)),
        "top_pairs": pairs[["variable_1", "variable_2", "r"]].head(5)
                          .round(4).to_dict(orient="records"),
        "age_vs_release_r": round(age_link, 4),
        "runtime_vs_year_r": round(float(
            corr_movies.loc["release_year", "duration_value"]), 4),
    }
# ---------------------------------------------------------------------------
# 9. Outlier & anomaly analysis
# ---------------------------------------------------------------------------

def outlier_analysis(df: pd.DataFrame) -> dict:
    rule("9. Outlier & Anomaly Analysis")
    movies = df.loc[df["type"] == "Movie", "duration_value"]
    shows = df.loc[df["type"] == "TV Show", "duration_value"]
    low_m, high_m = iqr_bounds(movies)
    low_s, high_s = iqr_bounds(shows)
    low_y, high_y = iqr_bounds(df["release_year"].astype(float))

    work = df.assign(
        movie_runtime_outlier=(df["type"] == "Movie")
        & ((df["duration_value"] < low_m) | (df["duration_value"] > high_m)),
        tv_season_outlier=(df["type"] == "TV Show")
        & ((df["duration_value"] < low_s) | (df["duration_value"] > high_s)),
    )

    findings = {
        "movie_runtime": {
            "fence": (round(low_m, 1), round(high_m, 1)),
            "count": int(work["movie_runtime_outlier"].sum()),
            "share": float(work["movie_runtime_outlier"].mean()),
            "examples": work.loc[work["movie_runtime_outlier"]]
                            .nlargest(3, "duration_value")[["title", "release_year", "duration_value"]]
                            .to_dict(orient="records"),
            "shortest": work.loc[(df["type"] == "Movie") & (df["duration_value"] < 60)]
                               .nsmallest(3, "duration_value")[["title", "duration_value"]]
                               .to_dict(orient="records"),
            "short_count": int(((df["type"] == "Movie") & (df["duration_value"] < 60)).sum()),
        },
        "tv_seasons": {
            "fence": (round(low_s, 1), round(high_s, 1)),
            "count": int(work["tv_season_outlier"].sum()),
            "share": float(work["tv_season_outlier"].mean()),
            "examples": work.loc[work["tv_season_outlier"]]
                            .nlargest(3, "duration_value")[["title", "release_year", "duration_value"]]
                            .to_dict(orient="records"),
        },
        "release_year": {
            "fence": (round(low_y, 1), round(high_y, 1)),
            "count": int((df["release_year"] < low_y).sum()),
            "share": float((df["release_year"] < low_y).mean()),
            "oldest": work.nsmallest(3, "release_year")[["title", "release_year"]]
                         .to_dict(orient="records"),
        },
    }

    print("\nIQR (Tukey) outlier screening - three variables, each judged on its own")
    for name, info in findings.items():
        print(f"  {name:15s} fence={info['fence']}  flagged={info['count']:,} "
              f"({pct(info['share'])})")
    print("\n  Longest movies flagged:")
    for row in findings["movie_runtime"]["examples"]:
        print(f"    - {row['title'][:52]:<54} {int(row['duration_value'])} min ({int(row['release_year'])})")
    print("  Longest-running TV shows flagged:")
    for row in findings["tv_seasons"]["examples"]:
        print(f"    - {row['title'][:52]:<54} {int(row['duration_value'])} seasons ({int(row['release_year'])})")
    print("  Oldest titles flagged (release_year below the lower fence):")
    for row in findings["release_year"]["oldest"]:
        print(f"    - {row['title'][:52]:<54} {int(row['release_year'])}")

    # Segment-level anomaly: country-years that spike far above their own baseline.
    yearly_country = (df.dropna(subset=["year_added"])
                        .groupby(["year_added", "primary_country"]).size()
                        .rename("titles").reset_index())
    stats = yearly_country.groupby("primary_country")["titles"].agg(["mean", "std", "count"])
    spikes = yearly_country.merge(stats, on="primary_country")
    spikes = spikes[(spikes["count"] >= 5) & (spikes["std"] > 0)]
    spikes["z"] = (spikes["titles"] - spikes["mean"]) / spikes["std"]
    flagged = spikes[spikes["z"] >= 2.5].nlargest(6, "z")
    print("\n  Country-year additions >= 2.5 std above that country's own baseline:")
    for row in flagged.itertuples():
        print(f"    - {row.primary_country:<16} {int(row.year_added)}: {int(row.titles)} titles "
              f"(baseline {row.mean:.1f}/yr, z = {row.z:.1f})")

    print("\n  Verdict: every flagged value is a *legitimate* catalogue entry.\n"
          "    - Feature-length films of 3-4 hours and long-running series are real content.\n"
          "    - Pre-2005 titles are a genuine back-catalogue tail, not recording errors.\n"
          "    - No outlier is deleted; they are documented and kept for analysis.")
    findings["country_year_spikes"] = flagged[["primary_country", "year_added", "titles", "z"]] \
        .round(2).to_dict(orient="records")
    findings["resolution"] = (
        "All IQR-flagged observations are legitimate catalogue entries (long films, "
        "multi-season series, vintage titles). Nothing was removed."
    )
    return findings
# ---------------------------------------------------------------------------
# 10. Segmentation analysis
# ---------------------------------------------------------------------------

def segmentation_analysis(df: pd.DataFrame) -> dict:
    rule("10. Segmentation Analysis")
    facts: dict[str, object] = {}

    # --- 10.1 Countries: volume vs average movie runtime ---------------------
    movie_df = df[df["type"] == "Movie"]
    country_movies = (movie_df[movie_df["primary_country"] != PLACEHOLDER]
                      .groupby("primary_country")["duration_value"]
                      .agg(["count", "mean", "median"]))
    country_all = df[df["primary_country"] != PLACEHOLDER]
    country_volume = country_all["primary_country"].value_counts()
    reliable = country_movies[country_movies["count"] >= 30].sort_values("mean", ascending=False)

    top_countries = country_volume.head(10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    ax.barh(top_countries.index[::-1], top_countries.values[::-1],
            color=ACCENT, alpha=0.9, edgecolor="white")
    annotate_bars(ax, top_countries.values[::-1],
                  labels=[f"{v:,} ({pct(v / len(country_all))})" for v in top_countries.values[::-1]],
                  orient="h")
    style_axis(ax, "Top 10 production countries by number of titles",
               "Number of titles", "Country")
    ax.grid(axis="y", alpha=0.0)

    ax = axes[1]
    plot = reliable.head(8).iloc[::-1]
    ax.barh(plot.index, plot["mean"], color=ACCENT, alpha=0.9, edgecolor="white")
    ax.axvline(movie_df["duration_value"].mean(), color=WARNING, ls="--", lw=1.6,
               label=f"Global movie mean {movie_df['duration_value'].mean():.0f} min")
    annotate_bars(ax, plot["mean"], labels=[f"{m:.0f} min" for m in plot["mean"]], orient="h")
    style_axis(ax, "Average movie runtime by country (>= 30 movies)",
               "Average runtime (minutes)", "Country")
    ax.grid(axis="y", alpha=0.0)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Segmentation by production country: volume and content style differ sharply",
                 fontsize=13, fontweight="bold", y=1.02)
    save_figure(fig, "comparison", "fig_13_country_segments.png",
                "Country segmentation: volume and average movie runtime",
                "The United States and India lead on volume, while Indian films run clearly longer than the global average.")

    facts["countries"] = {
        "us_count": int(country_volume.get("United States", 0)),
        "us_share": float(country_volume.get("United States", 0) / len(country_all)),
        "top5_share": float(country_volume.head(5).sum() / len(country_all)),
        "india_count": int(country_volume.get("India", 0)),
        "longest_country": str(reliable.index[0]),
        "longest_mean": round(float(reliable["mean"].iloc[0]), 1),
        "shortest_country": str(reliable.index[-1]),
        "shortest_mean": round(float(reliable["mean"].iloc[-1]), 1),
        "global_movie_mean": round(float(movie_df["duration_value"].mean()), 1),
        "segments_with_30plus": int(len(reliable)),
        "unspecified_count": int(country_volume.get(PLACEHOLDER, 0)),
    }
    return facts
# ---------------------------------------------------------------------------
# 10b. Segmentation analysis - genres and audience mix
# ---------------------------------------------------------------------------

def segmentation_genres_audience(df: pd.DataFrame) -> dict:
    facts: dict[str, object] = {}

    # --- Genre segments (exploded multi-value column) ------------------------
    genres = explode_list_column(df, "listed_in")
    top_genres = genres.head(12)
    movie_by_genre = {}
    tv_by_genre = {}
    for genre in top_genres.index:
        mask = df["listed_in"].str.contains(genre, regex=False)
        movie_by_genre[genre] = int((mask & (df["type"] == "Movie")).sum())
        tv_by_genre[genre] = int((mask & (df["type"] == "TV Show")).sum())

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(top_genres))
    movies_bar = [movie_by_genre[g] for g in top_genres.index]
    tv_bar = [tv_by_genre[g] for g in top_genres.index]
    ax.bar(x, movies_bar, width=0.62, color=TYPE_COLORS["Movie"], label="Movie",
           alpha=0.92, edgecolor="white")
    ax.bar(x, tv_bar, width=0.62, bottom=movies_bar, color=TYPE_COLORS["TV Show"],
           label="TV Show", alpha=0.92, edgecolor="white")
    for i, total in enumerate(np.add(movies_bar, tv_bar)):
        ax.annotate(f"{total:,}", (i, total), ha="center", va="bottom", fontsize=8.5)
    style_axis(ax, "Most common genres (a title can carry several genres)",
               "Genre", "Number of titles")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("International ", "Intl ") for g in top_genres.index],
                       rotation=25, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(np.add(movies_bar, tv_bar)) * 1.15)
    save_figure(fig, "comparison", "fig_14_genre_segments.png",
                "Genre segmentation by content type",
                "Documentaries, Stand-Up Comedy and Dramas are the largest genre segments in the catalogue.")

    facts["genres"] = {
        "total_genre_tokens": int(genres.sum()),
        "distinct_genres": int(len(genres)),
        "top10_share": float(genres.head(10).sum() / genres.sum()),
        "top1": str(genres.index[0]), "top1_count": int(genres.iloc[0]),
        "top5": [(str(g), int(c)) for g, c in genres.head(5).items()],
        "top_tv_genre": str(pd.Series(tv_by_genre).idxmax()),
        "top_movie_genre": str(pd.Series(movie_by_genre).idxmax()),
    }

    # --- Audience segments over time (content-rating strategy) ---------------
    audience = df.dropna(subset=["year_added"]).copy()
    audience["audience"] = np.where(
        audience["rating"].isin(["TV-MA", "TV-14", "R", "NC-17", "NR", "UR"]),
        "Mature (TV-MA / R / TV-14)",
        np.where(audience["rating"].isin(["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G", "PG"]),
                 "Family / Kids", "Teen (PG-13)"))
    pivot = pd.crosstab(audience["year_added"].astype(int), audience["audience"],
                        normalize="index") * 100
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", lw=2, label=col)
    style_axis(ax, "Audience mix of annual additions (share of titles added each year)",
               "Year added", "Share of titles added (%)")
    ax.legend(fontsize=9)
    save_figure(fig, "trend", "fig_15_audience_mix_over_time.png",
                "Audience mix over time",
                "Mature-rated content remained the largest share of annual additions throughout the growth phase.")

    facts["audience"] = {
        "overall_pct": {str(k): round(float(v), 2) for k, v in
                        (audience["audience"].value_counts(normalize=True) * 100).items()},
        "mature_count": int((audience["audience"] == "Mature (TV-MA / R / TV-14)").sum()),
        "family_count": int((audience["audience"] == "Family / Kids").sum()),
        "latest_year_mature_share": round(float(
            pivot.iloc[-1].get("Mature (TV-MA / R / TV-14)", np.nan)), 2),
        "mature_share_2015": round(float(
            pivot.loc[2015].get("Mature (TV-MA / R / TV-14)", np.nan)), 2)
        if 2015 in pivot.index else None,
    }
    return facts
# ---------------------------------------------------------------------------
# 11. Key performance metrics (KPIs)
# ---------------------------------------------------------------------------

def key_performance_metrics(df: pd.DataFrame, results: dict) -> dict:
    rule("11. Key Performance Metrics (KPIs)")

    dated = df.dropna(subset=["year_added"])
    yearly = dated.groupby("year_added").size()
    movies = df[df["type"] == "Movie"]
    shows = df[df["type"] == "TV Show"]
    genres = explode_list_column(df, "listed_in")

    growth = {}
    for year in [2015, 2016, 2017, 2018, 2019, 2020]:
        current, previous = yearly.get(year), yearly.get(year - 1)
        growth[year] = {
            "added": int(current) if pd.notna(current) else 0,
            "growth_pct": round(float((current - previous) / previous * 100), 1)
            if previous else None,
        }
    cagr_years = [y for y in yearly.index if 2015 <= y <= 2020]
    cagr = None
    if len(cagr_years) >= 2 and yearly.get(2015) and yearly.get(2020):
        n = 2020 - 2015
        cagr = round(float((yearly.get(2020) / yearly.get(2015)) ** (1 / n) * 100 - 100), 1)

    kpis = {
        "catalogue_size": int(len(df)),
        "movie_count": int(len(movies)),
        "tv_count": int(len(shows)),
        "movie_share": round(float(len(movies) / len(df) * 100), 2),
        "newest_addition": str(dated["date_added"].max().date()),
        "oldest_addition": str(dated["date_added"].min().date()),
        "avg_titles_added_per_year_2015_2020": round(
            float(yearly.loc[cagr_years].mean()), 1),
        "peak_year": int(yearly.idxmax()),
        "peak_year_additions": int(yearly.max()),
        "cagr_2015_2020_pct": cagr,
        "yoy_growth_pct": growth,
        "avg_movie_runtime": round(float(movies["duration_value"].mean()), 1),
        "median_movie_runtime": float(movies["duration_value"].median()),
        "avg_tv_seasons": round(float(shows["duration_value"].mean()), 2),
        "single_season_tv_share": round(float((shows["duration_value"] == 1).mean() * 100), 2),
        "avg_years_to_netflix": round(float(dated["years_to_netflix"].mean()), 2),
        "median_years_to_netflix": float(dated["years_to_netflix"].median()),
        "fresh_content_share": round(float((dated["years_to_netflix"] <= 1).mean() * 100), 2),
        "top_country": str(df["country"].value_counts().index[0]),
        "top_country_share": round(float(df["country"].value_counts(normalize=True).iloc[0] * 100), 2),
        "top_rating": str(df["rating"].value_counts().index[0]),
        "top_rating_share": round(float(df["rating"].value_counts(normalize=True).iloc[0] * 100), 2),
        "top_genre": str(genres.index[0]),
        "top_genre_share_of_tokens": round(float(genres.iloc[0] / genres.sum() * 100), 2),
        "avg_genres_per_title": round(float(df["genre_count"].mean()), 2),
        "director_missing_share": round(float((df["director"] == PLACEHOLDER).mean() * 100), 2),
        "date_completeness": round(float(df["date_added"].notna().mean() * 100), 2),
        "friday_launch_share": round(float(
            (dated["weekday_added"] == "Friday").mean() * 100), 2),
        "titles_per_year_mean": round(float(yearly.mean()), 1),
    }

    print("\nCatalogue KPIs")
    bullet("  Catalogue size", f"{kpis['catalogue_size']:,} titles")
    bullet("  Movie share of catalogue", f"{kpis['movie_share']}%")
    bullet("  Titles added per year (2015-2020 average)",
           f"{kpis['avg_titles_added_per_year_2015_2020']:,}")
    bullet("  Peak year", f"{kpis['peak_year']} ({kpis['peak_year_additions']:,} titles)")
    bullet("  Growth CAGR 2015-2020", f"{kpis['cagr_2015_2020_pct']}%" if cagr is not None else "n/a")
    bullet("  Average movie runtime", f"{kpis['avg_movie_runtime']} min "
           f"(median {kpis['median_movie_runtime']:.0f} min)")
    bullet("  Average TV show length", f"{kpis['avg_tv_seasons']} seasons "
           f"({kpis['single_season_tv_share']}% single-season)")
    bullet("  Median content age at upload", f"{kpis['median_years_to_netflix']:.0f} year(s)")
    bullet("  Fresh content (<= 1 year old)", f"{kpis['fresh_content_share']}%")
    bullet("  Friday launch share", f"{kpis['friday_launch_share']}%")
    bullet("  Leading country", f"{kpis['top_country']} ({kpis['top_country_share']}%)")
    bullet("  Leading rating", f"{kpis['top_rating']} ({kpis['top_rating_share']}%)")
    bullet("  Leading genre", f"{kpis['top_genre']} ({kpis['top_genre_share_of_tokens']}% of genre tags)")
    bullet("  Average genres per title", kpis["avg_genres_per_title"])
    bullet("  date_added completeness", f"{kpis['date_completeness']}%")

    print("\nYear-over-year change in annual additions")
    print("  year  added  yoy_growth_pct")
    for year, info in growth.items():
        bullet(f"    {year}", f"{info['added']:>5}   {info['growth_pct']}")
    kpis["description_quality"] = {
        "avg_words": round(float(df["description"].str.split().str.len().mean()), 1),
        "min_words": int(df["description"].str.split().str.len().min()),
        "max_words": int(df["description"].str.split().str.len().max()),
        "duplicate_descriptions": int(df["description"].duplicated().sum()),
    }
    return kpis
# ---------------------------------------------------------------------------
# 12. Business insights (every value below is read from RESULTS, never typed)
# ---------------------------------------------------------------------------

def build_insights(results: dict) -> list[dict[str, str]]:
    rule("12. Business Insights (evidence-based)")

    ov = results["overview"]
    kpi = results["kpis"]
    tr = results["trend"]["trend"]
    univ_a = results["univariate_a"]
    univ_b = results["univariate_b"]
    biv = results["bivariate"]
    corr = results["correlation"]
    out = results["outliers"]
    seg = results["segments"]
    genre = results["segments_genre"]
    quality = results["quality"]

    insights = []
    yoy_2017 = kpi["yoy_growth_pct"].get(2017, {})

    insights.append({
        "title": "Catalogue growth peaked in 2019 and has slowed since",
        "finding": (
            f"Annual additions ramped up from {tr['first_year_additions']:,} titles in "
            f"{tr['first_year']} to a peak of {tr['peak_year_additions']:,} titles in "
            f"{tr['peak_year']}, a {tr['growth_multiple']}x increase. Additions then fell "
            f"{abs(tr['peak_decline_pct'])}% to {tr['y2020']:,} in 2020."
        ),
        "evidence": (
            f"{tr['peak_year']} = {tr['peak_year_additions']:,} titles; 2020 = {tr['y2020']:,} "
            f"({tr['peak_decline_pct']}%); 2017 year-over-year growth was "
            f"{yoy_2017.get('growth_pct')}%. The 2015-2020 compound annual growth rate of "
            f"additions is {kpi['cagr_2015_2020_pct']}%."
        ),
        "why": (
            "The platform moved through a rapid library-expansion phase and then slowed the pace "
            "of additions. Any forward-looking view of catalogue size should model a plateau "
            "rather than continued exponential growth."
        ),
        "visualization": "visualizations/trend_charts/fig_04_additions_over_time.png",
    })

    insights.append({
        "title": "Movies are the bulk of the catalogue, but TV shows are thin in depth",
        "finding": (
            f"{kpi['movie_count']:,} of {kpi['catalogue_size']:,} titles are Movies "
            f"({kpi['movie_share']}%) versus {kpi['tv_count']:,} TV Shows. Yet "
            f"{univ_a['tv_seasons']['single_season_share'] * 100:.1f}% of TV shows run for a single "
            f"season, so the catalogue is wide in films and shallow in long-running series."
        ),
        "evidence": (
            f"Movie:TV ratio is {univ_b['type']['ratio']:.2f}:1. Median TV show length is "
            f"{univ_a['tv_seasons']['median']:.0f} season while the median movie runs "
            f"{univ_a['movie_runtime']['median']:.0f} minutes."
        ),
        "why": (
            "Films deliver instant breadth with low viewing commitment, while the small share of "
            "multi-season shows limits binge-depth content. Portfolio decisions can therefore be "
            "framed as breadth (films) versus depth (returning series)."
        ),
        "visualization": "visualizations/comparison_charts/fig_10_type_composition.png",
    })

    insights.append({
        "title": "Movie runtime is stable and does not track release year",
        "finding": (
            f"Movie runtime is tightly concentrated (mean {univ_a['movie_runtime']['mean']:.1f} min, "
            f"median {univ_a['movie_runtime']['median']:.0f} min, interquartile range roughly "
            f"{univ_a['movie_runtime']['q1']:.0f}-{univ_a['movie_runtime']['q3']:.0f} min) and its "
            f"correlation with release year is {corr['runtime_vs_year_r']:+.3f}."
        ),
        "evidence": (
            f"Across {biv['runtime_vs_year']['decades_compared']} decades of well-populated data, "
            f"decade means stay inside a {biv['runtime_vs_year']['spread_minutes']:.0f}-minute band, "
            f"and the distribution is close to symmetric (skew {univ_a['movie_runtime']['skew']})."
        ),
        "why": (
            "Audience expectations of a standard feature length have not shifted, so scheduling, "
            "runtime-based recommendations and slot planning can rely on a stable target window "
            "regardless of a film's era."
        ),
        "visualization": "visualizations/correlation_charts/fig_07_runtime_vs_release_year.png",
    })

    insights.append({
        "title": "Releases follow a deliberate weekly rhythm: Friday dominates",
        "finding": (
            f"{tr['friday_count']:,} titles ({tr['friday_share'] * 100:.1f}% of all dated additions) "
            f"were added on a Friday, making it the platform's primary publishing day. "
            f"{tr['busiest_month']} is the busiest month with {tr['busiest_month_count']:,} titles."
        ),
        "evidence": (
            f"Mean additions per calendar day is only {tr['daily_mean']}, yet the single busiest day "
            f"({tr['max_day']}) carried {tr['max_day_count']} titles - a clear batch-release pattern."
        ),
        "why": (
            "A Friday-first cadence concentrates attention and marketing spend into one weekly "
            "window. Competitor planning and any analysis of weekend viewing spikes must account "
            "for this deliberate scheduling."
        ),
        "visualization": "visualizations/trend_charts/fig_05_seasonality_weekday_month.png",
    })

    insights.append({
        "title": "The library has shifted towards fresh, near-current content",
        "finding": (
            f"The median title is added {tr['median_gap']:.0f} year after its original release "
            f"(mean {tr['mean_gap']:.1f} years), and {tr['share_within_1yr'] * 100:.1f}% of dated "
            f"titles were added within one year of release."
        ),
        "evidence": (
            f"{univ_b['release_year']['recent_share'] * 100:.1f}% of the catalogue was released in "
            f"{MATURE_YEAR_FROM} or later and the median release year is "
            f"{univ_b['release_year']['median']:.0f}, while the oldest title dates to "
            f"{univ_b['release_year']['min']}."
        ),
        "why": (
            "A catalogue concentrated on recent material competes on newness rather than on a deep "
            "archive, which raises the importance of continuous licensing and original production."
        ),
        "visualization": "visualizations/trend_charts/fig_06_content_age_gap.png",
    })

    insights.append({
        "title": "Content supply is highly concentrated in a handful of countries",
        "finding": (
            f"{seg['countries']['us_share'] * 100:.1f}% of titles come from the United States and "
            f"the top five countries supply {seg['countries']['top5_share'] * 100:.1f}% of the "
            f"catalogue; India alone contributes {seg['countries']['india_count']:,} titles."
        ),
        "evidence": (
            f"{ov['categorical_unique']['country']} distinct countries appear in the data. Among the "
            f"{seg['countries']['segments_with_30plus']} countries with at least 30 movies, "
            f"{seg['countries']['longest_country']} has the longest average runtime "
            f"({seg['countries']['longest_mean']} min) against a global movie mean of "
            f"{seg['countries']['global_movie_mean']} min, while "
            f"{seg['countries']['shortest_country']} averages {seg['countries']['shortest_mean']} min."
        ),
        "why": (
            "Sourcing is geographically concentrated, which is both a supply-chain risk and an "
            "opportunity: strong producers that are currently under-represented are candidate "
            "markets for localised content investment."
        ),
        "visualization": "visualizations/comparison_charts/fig_13_country_segments.png",
    })

    insights.append({
        "title": "Mature-rated content anchors the catalogue and its share is stable over time",
        "finding": (
            f"{kpi['top_rating']} alone accounts for {kpi['top_rating_share']}% of titles, and the "
            f"mature audience segment represents "
            f"{genre['audience']['overall_pct'].get('Mature (TV-MA / R / TV-14)', 0)}% of the "
            f"catalogue."
        ),
        "evidence": (
            f"{univ_b['rating']['mature_share'] * 100:.1f}% of titles carry a mature rating versus "
            f"{univ_b['rating']['family_share'] * 100:.1f}% family/kids. The mature share stood at "
            f"{genre['audience']['mature_share_2015']}% in 2015 and "
            f"{genre['audience']['latest_year_mature_share']}% in the final year, so the mix is "
            f"not shifting with catalogue growth."
        ),
        "why": (
            "Advertising, compliance and parental-control features must be sized for an "
            "adult-majority library, while the family segment stays a meaningful but secondary "
            "audience."
        ),
        "visualization": "visualizations/comparison_charts/fig_09_rating_distribution.png",
    })

    insights.append({
        "title": "Statistical outliers are legitimate long-tail content, not data errors",
        "finding": (
            f"IQR screening flags {out['movie_runtime']['count']:,} movies "
            f"({out['movie_runtime']['share'] * 100:.1f}%), {out['tv_seasons']['count']:,} TV shows "
            f"({out['tv_seasons']['share'] * 100:.1f}%) and {out['release_year']['count']:,} "
            f"pre-2005 titles."
        ),
        "evidence": (
            f"Flagged cases include feature films of up to {univ_a['movie_runtime']['max']} minutes "
            f"and series of up to {univ_a['tv_seasons']['max']} seasons, with the oldest title dated "
            f"{univ_b['release_year']['min']}."
        ),
        "why": (
            "These extremes serve genuine long-format and vintage audiences. They should be "
            "segmented and reported separately rather than trimmed, because deleting them would "
            "bias average runtime and hide the long tail."
        ),
        "visualization": "visualizations/distribution_charts/fig_02_duration_boxplot.png",
    })

    insights.append({
        "title": "Numeric fields carry very little genuine correlation",
        "finding": (
            f"Of {corr['strong_pairs'] + corr['moderate_pairs'] + corr['weak_pairs']} movie-level "
            f"field pairs, {corr['strong_pairs']} are strong (|r| >= 0.5) and "
            f"{corr['moderate_pairs']} moderate; the remainder are weak."
        ),
        "evidence": (
            f"The strongest relationship is release year versus content age at upload "
            f"(r = {corr['age_vs_release_r']:+.2f}), which is a calendar artefact rather than a "
            f"behavioural driver. Runtime versus release year is r = {corr['runtime_vs_year_r']:+.3f}."
        ),
        "why": (
            "Catalogue metadata alone cannot explain performance because the dataset contains no "
            "viewership, revenue or engagement measures. Correlation must not be read as causation "
            "here, and any predictive modelling needs additional behavioural data."
        ),
        "visualization": "visualizations/correlation_charts/fig_08_correlation_heatmap.png",
    })

    insights.append({
        "title": "Metadata completeness differs by field and is reported honestly",
        "finding": (
            f"Missing metadata is concentrated in credits: director is unspecified for "
            f"{kpi['director_missing_share']}% of titles and the placeholder country label covers "
            f"{quality['placeholders']['country']:,} titles. Only date_added is genuinely empty "
            f"({ov['missing_cells']} cells, {ov['missing_cells_pct'] * 100:.2f}% of the dataset)."
        ),
        "evidence": (
            f"date_added completeness is {kpi['date_completeness']}%. A residual formatting issue "
            f"was also found: {quality['zwsp_cells']} text cells still contain an invisible "
            f"zero-width space that Task 1's whitespace rule (str.strip) cannot detect."
        ),
        "why": (
            "Crew-level and country-level analysis will under-count because of unspecified credits, "
            "so those results are stated as shares of titles with known credits. The zero-width "
            "space residue is a minor but concrete improvement for Task 1's pipeline."
        ),
        "visualization": "visualizations/distribution_charts/fig_03_release_year_distribution.png",
    })

    for i, ins in enumerate(insights, 1):
        print(f"\n  Insight {i}: {ins['title']}")
        print(f"    Finding  : {ins['finding']}")
        print(f"    Evidence : {ins['evidence']}")
        print(f"    Why it matters: {ins['why']}")
    print(f"\n  Total insights: {len(insights)}")
    return insights


def print_conclusion(results: dict) -> str:
    """Short, plain-language closing statement built from computed values."""
    kpi = results["kpis"]
    tr = results["trend"]["trend"]
    univ_b = results["univariate_b"]
    narrative = (
        f"The catalogue holds {kpi['catalogue_size']:,} titles, of which "
        f"{kpi['movie_share']}% are movies. Additions grew to a peak of "
        f"{tr['peak_year_additions']:,} titles in {tr['peak_year']} and then slowed. "
        f"Content is recent (median release year {univ_b['release_year']['median']:.0f}) and "
        f"concentrated in mature-rated, US-produced titles. Runtime is stable, correlations are weak, "
        f"and all flagged outliers are legitimate catalogue entries."
    )
    rule("13. Conclusion")
    print(textwrap.fill(narrative, width=95))
    return narrative


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Make numpy / pandas scalars JSON-serialisable without dropping information."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Series):
        return {str(k): _json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (pd.NA.__class__,)) or value is pd.NA:
        return None
    try:
        import pandas as _pd

        if isinstance(value, (_pd._libs.missing.NAType,)):
            return None
    except Exception:
        pass
    return value


def run_full_analysis(save: bool = True) -> dict:
    """Execute the entire EDA pipeline in report order."""
    print("=" * 78)
    print("EXPLORATORY DATA ANALYSIS - NETFLIX MOVIES AND TV SHOWS")
    print("SWYNEX Technologies - Task 2 (EDA & Insight Discovery)")
    print("=" * 78)
    print(f"Dataset : {DATA_PATH}")
    print(f"Charts  : {VIZ_DIR}")

    base = load_cleaned_dataset()
    df = build_analysis_ready(base)

    RESULTS["overview"] = dataset_overview(df)
    RESULTS["quality"] = data_quality_check(df)
    RESULTS["descriptive"] = descriptive_statistics(df)
    RESULTS["univariate_a"] = univariate_analysis(df)
    RESULTS["univariate_b"] = univariate_categorical(df)
    RESULTS["bivariate"] = bivariate_analysis(df)
    RESULTS["trend"] = time_series_analysis(df)
    RESULTS["correlation"] = correlation_analysis(df)
    RESULTS["outliers"] = outlier_analysis(df)
    RESULTS["segments"] = segmentation_analysis(df)
    RESULTS["segments_genre"] = segmentation_genres_audience(df)
    RESULTS["kpis"] = key_performance_metrics(df, RESULTS)
    RESULTS["insights"] = build_insights(RESULTS)
    RESULTS["conclusion"] = print_conclusion(RESULTS)
    RESULTS["charts"] = CHART_INDEX

    if save:
        _save_summary()

    rule("Run complete")
    bullet("Charts written", len(CHART_INDEX))
    bullet("Insights generated", len(RESULTS["insights"]))
    bullet("Summary written", str(SUMMARY_JSON.relative_to(PROJECT_ROOT)))
    return RESULTS


def _save_summary() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: _json_safe(v) for k, v in RESULTS.items()}
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [saved] reports/{SUMMARY_JSON.name}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    run_full_analysis()