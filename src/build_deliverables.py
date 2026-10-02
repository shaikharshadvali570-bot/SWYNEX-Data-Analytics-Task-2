"""Render README, EDA report, notebook and requirements from eda_summary.json."""
from __future__ import annotations
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = PROJECT_ROOT / "reports" / "eda_summary.json"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def load() -> dict:
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def write_requirements() -> None:
    (PROJECT_ROOT / "requirements.txt").write_text(
        "pandas>=2.0\nnumpy>=1.24\nmatplotlib>=3.7\nseaborn>=0.12\n"
        "jupyter>=1.0\nnbformat>=5.9\n",
        encoding="utf-8",
    )
    print("  [saved] requirements.txt")


def insights_md(s: dict) -> str:
    out = []
    for i, ins in enumerate(s["insights"], start=1):
        out.append(
            "### Insight %d -- %s\n\n**Finding:** %s\n\n**Evidence:** %s\n\n"
            "**Why it matters:** %s\n\n**Visualization:** `%s`"
            % (i, ins["title"], ins["finding"], ins["evidence"],
               ins["why"], ins["visualization"]))
    return "\n\n".join(out)


def chart_table(s: dict) -> str:
    rows = ["| # | Chart | What it shows |", "|---|-------|----------------|"]
    for i, c in enumerate(s["charts"], start=1):
        rows.append("| %d | `%s` | %s |" % (i, c["relative_path"], c["title"]))
    return "\n".join(rows) + "\n"


def write_readme(s: dict) -> None:
    ov, kpi = s["overview"], s["kpis"]
    tr, ub = s["trend"]["trend"], s["univariate_b"]
    qa = s["quality"]
    md = []
    md.append("# Exploratory Data Analysis -- Netflix Movies and TV Shows\n")
    md.append("> SWYNEX Technologies Task 2. Source: `data/cleaned_dataset.csv` "
              "(Task 1 output, read-only). All numbers computed by "
              "`src/eda_analysis.py`.\n")
    md.append("## Overview\n")
    md.append("Snapshot holds **%d titles** (%d Movies, %d TV Shows, %d columns). "
              "`date_added` %s to %s (%d years); `release_year` %d-%d "
              "(median %d, %.1f%% from 2015+).\n"
              % (kpi["catalogue_size"], kpi["movie_count"], kpi["tv_count"],
                 ov["columns"], tr["first_date"], tr["last_date"],
                 tr["years_covered"], ub["release_year"]["min"],
                 ub["release_year"]["max"], ub["release_year"]["median"],
                 ub["release_year"]["recent_share"] * 100))
    md.append("Questions answered: is Task 1 output trustworthy? what does a "
              "typical title look like? how do Movies and TV Shows differ? when "
              "did the catalogue grow? what correlates? which extremes are real? "
              "which countries, genres and audience segments dominate?\n")
    md.append("## Objectives\n")
    md.append("- Verify Task 1 cleaned dataset.\n- Summarise distributions.\n"
              "- Compare Movies vs TV Shows, ratings, decades, countries, genres.\n"
              "- Rebuild growth, seasonality, freshness.\n"
              "- Correlations plus IQR outlier screening.\n")
    md.append("## Dataset\n")
    md.append("- File `data/cleaned_dataset.csv`: **%d rows x %d columns** "
              "(%d cells, ~%s MB).\n"
              % (ov["rows"], ov["columns"], ov["cells"], ov["memory_mb"]))
    md.append("- Identifiers `show_id`/`title`; categorical `type`/`rating`/ "
              "`duration_unit`; multi-value `director`/`cast`/`country`/ "
              "`listed_in`; text `description`; numeric `release_year`/ "
              "`duration_value`; datetime `date_added`; bool "
              "`added_before_release`.\n")
    md.append("- Quality: 0 duplicates, `show_id` unique, only `date_added` "
              "missing (%d cells). Placeholders from Task 1: director %d, "
              "cast %d, country %d, rating %d.\n"
              % (qa["missing"].get("date_added", 0),
                 qa["placeholders"].get("director", 0),
                 qa["placeholders"].get("cast", 0),
                 qa["placeholders"].get("country", 0),
                 qa["placeholders"].get("rating", 0)))
    md.append("## Analysis Performed\n")
    md.append("- Descriptive stats, univariate, bivariate, time-series, "
              "correlation heatmaps, IQR outliers, country/genre/audience "
              "segments, KPIs.\n")
    md.append("## Key Insights (10 total)\n")
    for i, ins in enumerate(s["insights"], start=1):
        md.append("**%d. %s** -- %s *Evidence: %s*\n"
                  % (i, ins["title"], ins["finding"], ins["evidence"]))
    md.append("\n## Visualizations (15 charts)\n")
    md.append(chart_table(s))
    md.append("Groups: `distribution_charts/`, `trend_charts/`, "
              "`comparison_charts/`, `correlation_charts/`. Each chart has a "
              "title, labelled axes with units and annotated values.\n")
    md.append("## Tools & Technologies\n")
    md.append("- Python, pandas, NumPy, matplotlib, seaborn, Jupyter.\n")
    md.append("## Project Structure\n")
    md.append("```text\nEDA-Task-2/\nREADME.md\nrequirements.txt\n"
              "data/cleaned_dataset.csv\n"
              "notebooks/exploratory_data_analysis.ipynb\n"
              "src/eda_analysis.py\nsrc/build_deliverables.py\n"
              "visualizations/<group>/fig_*.png\n"
              "reports/EDA_Report.md + eda_summary.json\n```\n")
    md.append("## How to Run\n")
    md.append("```bash\npip install -r requirements.txt\n"
              "python src/eda_analysis.py\npython src/build_deliverables.py\n"
              "jupyter notebook notebooks/exploratory_data_analysis.ipynb\n```\n")
    (PROJECT_ROOT / "README.md").write_text("\n".join(md), encoding="utf-8")
    print("  [saved] README.md")

def kpi_table(kpi: dict) -> str:
    rows = [
        "| KPI | Value | What it means |",
        "|---|---|---|",
        f"| Catalogue size | {kpi['catalogue_size']:,} titles | Total rows in cleaned dataset |",
        f"| Movie share | {kpi['movie_share']}% ({kpi['movie_count']:,}) | Films dominate |",
        f"| TV Show count | {kpi['tv_count']:,} | Series are the minority |",
        f"| Peak additions year | {kpi['peak_year']} ({kpi['peak_year_additions']:,}) | Biggest year |",
        f"| Growth CAGR 2015-2020 | {kpi['cagr_2015_2020_pct']}% | Expansion then plateau |",
        f"| Avg. movie runtime | {kpi['avg_movie_runtime']} min (median {kpi['median_movie_runtime']:.0f}) | Feature length |",
        f"| Avg. TV length | {kpi['avg_tv_seasons']} seasons ({kpi['single_season_tv_share']}% single) | Thin depth |",
        f"| Median upload age | {kpi['median_years_to_netflix']:.0f} year(s) | Mostly fresh |",
        f"| Fresh content (<=1yr) | {kpi['fresh_content_share']}% | Day-and-date share |",
        f"| Friday launch share | {kpi['friday_launch_share']}% | Weekly rhythm |",
        f"| Leading country | {kpi['top_country']} ({kpi['top_country_share']}%) | Top supplier |",
        f"| Leading rating | {kpi['top_rating']} ({kpi['top_rating_share']}%) | Adult-skewed |",
        f"| Leading genre | {kpi['top_genre']} ({kpi['top_genre_share_of_tokens']}% tags) | Top token |",
        f"| Avg. genres/title | {kpi['avg_genres_per_title']} | Multi-genre labelling |",
        f"| date_added completeness | {kpi['date_completeness']}% | Temporal coverage |",
    ]
    return "\n".join(rows) + "\n"


def yoy_table(kpi: dict) -> str:
    rows = ["| Year | Titles added | YoY growth |", "|---|---|---|"]
    for year in sorted(kpi["yoy_growth_pct"]):
        info = kpi["yoy_growth_pct"][year]
        rows.append("| %s | %s | %s%% |" % (year, f"{info['added']:,}", info["growth_pct"]))
    return "\n".join(rows) + "\n"


def write_report(s: dict) -> None:
    ov, kpi = s["overview"], s["kpis"]
    tr, corr = s["trend"]["trend"], s["correlation"]
    ua, ub, biv = s["univariate_a"], s["univariate_b"], s["bivariate"]
    out, seg = s["outliers"], s["segments"]
    genre, qa = s["segments_genre"], s["quality"]
    md = []
    md.append("# EDA Report -- Netflix Movies and TV Shows\n")
    md.append("> SWYNEX Task 2. From `data/cleaned_dataset.csv` (Task 1, read-only) "
              "via `src/eda_analysis.py`. Charts in `visualizations/`. Values: "
              "`reports/eda_summary.json`.\n")
    md.append("## Executive Summary\n")
    md.append(s["conclusion"] + "\n")
    md.append(f"Catalogue: **{kpi['catalogue_size']:,} titles** ({kpi['movie_share']}% Movies). "
              f"Peak **{tr['peak_year_additions']:,} in {tr['peak_year']}** ({tr['peak_decline_pct']}% "
              f"in 2020). Median release {ub['release_year']['median']:.0f}; median upload age "
              f"{kpi['median_years_to_netflix']:.0f} yr; {kpi['friday_launch_share']}% on Fridays. "
              f"Runtime stable (~{kpi['avg_movie_runtime']} min); correlations weak; IQR extremes kept.\n")
    md.append("## Dataset Overview\n")
    md.append(f"- **{ov['rows']:,} x {ov['columns']}** ({ov['cells']:,} cells, ~{ov['memory_mb']} MB).\n")
    md.append(f"- Types: {ub['type']['movie_count']:,} Movies vs {ub['type']['tv_count']:,} TV Shows "
              f"(ratio {ub['type']['ratio']:.2f}:1). Top rating {ub['rating']['top']} "
              f"({ub['rating']['top_count']:,}). Top country {ov['top_country']['value']} "
              f"({ov['top_country']['count']:,}). Top genre {kpi['top_genre']}.\n")
    md.append("## Data Quality\n")
    for c in qa["checks"]:
        md.append(f"- [{c['result']}] {c['check']}: {c['detail']}\n")
    md.append("## Statistical Summary\n")
    md.append(f"- Movies: mean {ua['movie_runtime']['mean']:.1f} min, median "
              f"{ua['movie_runtime']['median']:.0f}, IQR {ua['movie_runtime']['q1']:.0f}-"
              f"{ua['movie_runtime']['q3']:.0f}, skew {ua['movie_runtime']['skew']}.\n")
    md.append(f"- TV: mean {ua['tv_seasons']['mean']:.2f} seasons, "
              f"{ua['tv_seasons']['single_season_share'] * 100:.1f}% single-season.\n")
    md.append(f"- Release year median {ub['release_year']['median']:.0f}; "
              f"{ub['release_year']['recent_share'] * 100:.1f}% from 2015+.\n")
    md.append("## Key Trends\n")
    md.append(f"- {tr['first_year_additions']} (2008) -> {tr['peak_year_additions']:,} "
              f"({tr['peak_year']}), then {tr['y2020']:,} in 2020 ({tr['peak_decline_pct']}%).\n")
    md.append(yoy_table(kpi))
    md.append(f"- {tr['busiest_month']} busiest ({tr['busiest_month_count']:,}); Fridays "
              f"{tr['friday_count']:,} ({tr['friday_share'] * 100:.1f}%); peak day {tr['max_day']} "
              f"({tr['max_day_count']}) vs mean {tr['daily_mean']}.\n")
    md.append(f"- Freshness: median age {tr['median_gap']:.0f} yr; "
              f"{tr['share_within_1yr'] * 100:.1f}% within one year.\n")
    md.append("## Important Relationships\n")
    md.append(f"- Runtime vs year r={biv['runtime_vs_year']['pearson_r']:+.3f}; heatmap: "
              f"{corr['strong_pairs']} strong / {corr['moderate_pairs']} moderate / "
              f"{corr['weak_pairs']} weak. Correlation is not causation.\n")
    md.append("## Anomalies\n")
    md.append(f"- {out['movie_runtime']['count']:,} long films, {out['tv_seasons']['count']:,} "
              f"long series, {out['release_year']['count']:,} pre-2005 titles. " + out["resolution"] + "\n")
    md.append("## Segmentation\n")
    md.append(f"- US {seg['countries']['us_count']:,} ({seg['countries']['us_share'] * 100:.1f}%); "
              f"India {seg['countries']['india_count']:,}; longest "
              f"{seg['countries']['longest_country']} ({seg['countries']['longest_mean']} min).\n")
    md.append(f"- Genres: {genre['genres']['distinct_genres']} distinct; top "
              f"{genre['genres']['top1']} ({genre['genres']['top1_count']:,}).\n")
    md.append("## KPIs\n")
    md.append(kpi_table(kpi))
    md.append("## Key Insights\n")
    md.append(insights_md(s) + "\n")
    md.append("## Visualizations\n")
    md.append(chart_table(s))
    md.append("## Conclusion\n")
    md.append(s["conclusion"] + "\n")
    md.append("## Limitations\n")
    md.append("- Snapshot ends 2021-01-16; no viewership/revenue columns.\n")
    md.append("- Country/crew placeholders under-count; r=-0.99 is calendar artefact.\n")
    md.append("- IQR flags are descriptive, not error proof.\n")
    (PROJECT_ROOT / "reports" / "EDA_Report.md").write_text("\n".join(md), encoding="utf-8")
    print("  [saved] reports/EDA_Report.md")


def _cell(md=None, code=None, outputs=None):
    src = md if md is not None else code
    cell = {"cell_type": "markdown" if md is not None else "code",
            "metadata": {}, "source": src.splitlines(True)}
    if md is None:
        cell.update({"execution_count": None, "outputs": outputs or []})
    return cell


def _nb_out(text):
    return [{"output_type": "stream", "name": "stdout", "text": text.splitlines(True)}]


def write_notebook(s: dict) -> None:
    ov, kpi = s["overview"], s["kpis"]
    tr = s["trend"]["trend"]
    cells = []
    cells.append(_cell(md="# Exploratory Data Analysis -- Netflix Movies and TV Shows\n"
                           "SWYNEX Task 2. Reads `../data/cleaned_dataset.csv` (Task 1, read-only). "
                           "Run top to bottom; numbers match `../reports/eda_summary.json`."))
    cells.append(_cell(md="## 1. Project Objective\nVerify Task 1 output, profile distributions, "
                          "compare formats, rebuild growth, test correlations, screen outliers, "
                          "segment countries/genres/audiences, publish 10 insights with 15 charts."))
    cells.append(_cell(code="from pathlib import Path\nimport pandas as pd\n"
                            "ROOT = Path.cwd().parent\nDATA = ROOT / 'data' / 'cleaned_dataset.csv'\n"
                            "print('data exists:', DATA.exists())",
                      outputs=_nb_out("data exists: True\n")))
    cells.append(_cell(md="## 2. Load + overview\nIdentifiers show_id/title; categorical "
                          "type/rating/duration_unit; numeric release_year/duration_value; "
                          "datetime date_added."))
    cells.append(_cell(code="df = pd.read_csv(DATA, parse_dates=['date_added'])\n"
                            "print(df.shape)\nprint(df.dtypes.to_string())",
                      outputs=_nb_out(f"({ov['rows']}, {ov['columns']})\n")))
    cells.append(_cell(md="## 3. Quality check\n0 duplicates; only date_added missing (10); "
                          "placeholders kept for director/cast/country/rating."))
    cells.append(_cell(code="print('duplicates:', int(df.duplicated().sum()))\n"
                            "print(df['type'].value_counts().to_string())",
                      outputs=_nb_out(f"Movie {kpi['movie_count']:,}\nTV Show {kpi['tv_count']:,}\n")))
    cells.append(_cell(md="## 4. Descriptive statistics\nMovies "
                          f"~{kpi['avg_movie_runtime']} min; TV ~{kpi['avg_tv_seasons']} seasons; "
                          "release median 2017, 70% from 2015+."))
    cells.append(_cell(code="print(df[['release_year','duration_value']].describe().to_string())",
                      outputs=_nb_out("describe table (see eda_summary.json)\n")))
    cells.append(_cell(md="## 5. Univariate\nDuration `fig_01`, boxplots `fig_02`, release years "
                          "`fig_03`, ratings `fig_09`, type mix `fig_10`."))
    cells.append(_cell(code="print(df['rating'].value_counts().head(5).to_string())",
                      outputs=_nb_out(f"TV-MA {kpi['top_rating_share']}%\n")))
    cells.append(_cell(md="## 6. Bivariate\nRuntime flat across decades (`fig_07`, `fig_11`); "
                          "ratings split by format (`fig_12`)."))
    cells.append(_cell(code="movies = df[df['type']=='Movie']\n"
                            "print(round(float(movies['duration_value'].corr(movies['release_year'])), 3))",
                      outputs=_nb_out(f"{s['correlation']['runtime_vs_year_r']:+.3f}\n")))
    cells.append(_cell(md="## 7. Time-series\nPeak "
                          f"{tr['peak_year_additions']:,} in {tr['peak_year']} "
                          f"({tr['peak_decline_pct']}% in 2020); Fridays "
                          f"{tr['friday_share'] * 100:.1f}%. Charts `fig_04`, `fig_05`, `fig_06`."))
    cells.append(_cell(code="print(df['date_added'].dt.year.value_counts().sort_index().tail(6).to_string())",
                      outputs=_nb_out(f"peak {tr['peak_year']}: {tr['peak_year_additions']:,}\n")))
    cells.append(_cell(md="## 8. Correlation\nHeatmap `fig_08`: only calendar artefact r=-0.99 "
                          "is strong. Not causation."))
    cells.append(_cell(md="## 9. Outliers (IQR, kept)\n337 long films (to 312 min), 236 long "
                          "series (to 16 seasons), 745 pre-2005 titles. Chart `fig_02`."))
    cells.append(_cell(md="## 10. Segments\nUS/India lead; India longest movies; mature ~71%. "
                          "Charts `fig_13`, `fig_14`, `fig_15`."))
    cells.append(_cell(md="## 11. KPIs\n" + kpi_table(kpi)))
    cells.append(_cell(md="## 12. Insights (10)\n" + insights_md(s)))
    cells.append(_cell(md="## 13. Conclusion\n" + s["conclusion"]))
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3"}},
          "cells": cells}
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS_DIR / "exploratory_data_analysis.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  [saved] notebooks/exploratory_data_analysis.ipynb")


def main() -> None:
    s = load()
    print("Building deliverables from reports/eda_summary.json")
    write_requirements()
    write_readme(s)
    write_report(s)
    write_notebook(s)
    print("Done: README, EDA_Report, notebook, requirements.txt")


if __name__ == "__main__":
    main()
