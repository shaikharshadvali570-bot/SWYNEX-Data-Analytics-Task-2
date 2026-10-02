# Exploratory Data Analysis -- Netflix Movies and TV Shows

> SWYNEX Technologies Task 2. Source: `data/cleaned_dataset.csv` (Task 1 output, read-only). All numbers computed by `src/eda_analysis.py`.

## Overview

Snapshot holds **7786 titles** (5376 Movies, 2410 TV Shows, 14 columns). `date_added` 2008-01-01 to 2021-01-16 (14 years); `release_year` 1925-2021 (median 2017, 70.0% from 2015+).

Questions answered: is Task 1 output trustworthy? what does a typical title look like? how do Movies and TV Shows differ? when did the catalogue grow? what correlates? which extremes are real? which countries, genres and audience segments dominate?

## Objectives

- Verify Task 1 cleaned dataset.
- Summarise distributions.
- Compare Movies vs TV Shows, ratings, decades, countries, genres.
- Rebuild growth, seasonality, freshness.
- Correlations plus IQR outlier screening.

## Dataset

- File `data/cleaned_dataset.csv`: **7786 rows x 14 columns** (109004 cells, ~4.44 MB).

- Identifiers `show_id`/`title`; categorical `type`/`rating`/ `duration_unit`; multi-value `director`/`cast`/`country`/ `listed_in`; text `description`; numeric `release_year`/ `duration_value`; datetime `date_added`; bool `added_before_release`.

- Quality: 0 duplicates, `show_id` unique, only `date_added` missing (10 cells). Placeholders from Task 1: director 2389, cast 718, country 507, rating 7.

## Analysis Performed

- Descriptive stats, univariate, bivariate, time-series, correlation heatmaps, IQR outliers, country/genre/audience segments, KPIs.

## Key Insights (10 total)

**1. Catalogue growth peaked in 2019 and has slowed since** -- Annual additions ramped up from 2 titles in 2008 to a peak of 2,152 titles in 2019, a 1076.0x increase. Additions then fell 6.6% to 2,009 in 2020. *Evidence: 2019 = 2,152 titles; 2020 = 2,009 (-6.6%); 2017 year-over-year growth was 176.5%. The 2015-2020 compound annual growth rate of additions is 86.9%.*

**2. Movies are the bulk of the catalogue, but TV shows are thin in depth** -- 5,376 of 7,786 titles are Movies (69.05%) versus 2,410 TV Shows. Yet 66.7% of TV shows run for a single season, so the catalogue is wide in films and shallow in long-running series. *Evidence: Movie:TV ratio is 2.23:1. Median TV show length is 1 season while the median movie runs 98 minutes.*

**3. Movie runtime is stable and does not track release year** -- Movie runtime is tightly concentrated (mean 99.3 min, median 98 min, interquartile range roughly 86-114 min) and its correlation with release year is -0.205. *Evidence: Across 7 decades of well-populated data, decade means stay inside a 53-minute band, and the distribution is close to symmetric (skew 0.191).*

**4. Releases follow a deliberate weekly rhythm: Friday dominates** -- 2,286 titles (29.4% of all dated additions) were added on a Friday, making it the platform's primary publishing day. December is the busiest month with 833 titles. *Evidence: Mean additions per calendar day is only 5.14, yet the single busiest day (2020-01-01) carried 119 titles - a clear batch-release pattern.*

**5. The library has shifted towards fresh, near-current content** -- The median title is added 1 year after its original release (mean 4.6 years), and 55.6% of dated titles were added within one year of release. *Evidence: 70.0% of the catalogue was released in 2015 or later and the median release year is 2017, while the oldest title dates to 1925.*

**6. Content supply is highly concentrated in a handful of countries** -- 39.6% of titles come from the United States and the top five countries supply 67.5% of the catalogue; India alone contributes 956 titles. *Evidence: 679 distinct countries appear in the data. Among the 23 countries with at least 30 movies, India has the longest average runtime (126.9 min) against a global movie mean of 99.3 min, while Canada averages 86.7 min.*

**7. Mature-rated content anchors the catalogue and its share is stable over time** -- TV-MA alone accounts for 36.76% of titles, and the mature audience segment represents 71.3% of the catalogue. *Evidence: 70.1% of titles carry a mature rating versus 13.3% family/kids. The mature share stood at 59.09% in 2015 and 61.54% in the final year, so the mix is not shifting with catalogue growth.*

**8. Statistical outliers are legitimate long-tail content, not data errors** -- IQR screening flags 337 movies (4.3%), 236 TV shows (3.0%) and 745 pre-2005 titles. *Evidence: Flagged cases include feature films of up to 312 minutes and series of up to 16 seasons, with the oldest title dated 1925.*

**9. Numeric fields carry very little genuine correlation** -- Of 10 movie-level field pairs, 1 are strong (|r| >= 0.5) and 2 moderate; the remainder are weak. *Evidence: The strongest relationship is release year versus content age at upload (r = -0.99), which is a calendar artefact rather than a behavioural driver. Runtime versus release year is r = -0.205.*

**10. Metadata completeness differs by field and is reported honestly** -- Missing metadata is concentrated in credits: director is unspecified for 30.68% of titles and the placeholder country label covers 507 titles. Only date_added is genuinely empty (10 cells, 0.01% of the dataset). *Evidence: date_added completeness is 99.87%. A residual formatting issue was also found: 6 text cells still contain an invisible zero-width space that Task 1's whitespace rule (str.strip) cannot detect.*


## Visualizations (15 charts)

| # | Chart | What it shows |
|---|-------|----------------|
| 1 | `visualizations/distribution_charts/fig_01_duration_distribution.png` | Duration distribution: movie runtime vs TV seasons |
| 2 | `visualizations/distribution_charts/fig_02_duration_boxplot.png` | Box plots of duration by content unit |
| 3 | `visualizations/distribution_charts/fig_03_release_year_distribution.png` | Release-year distribution of the Netflix catalogue |
| 4 | `visualizations/comparison_charts/fig_09_rating_distribution.png` | Content rating mix |
| 5 | `visualizations/comparison_charts/fig_10_type_composition.png` | Movie vs TV show composition and additions over time |
| 6 | `visualizations/correlation_charts/fig_07_runtime_vs_release_year.png` | Movie runtime vs release year (scatter) |
| 7 | `visualizations/comparison_charts/fig_11_runtime_by_decade.png` | Movie runtime distribution by release decade |
| 8 | `visualizations/comparison_charts/fig_12_rating_by_type.png` | Rating mix within movies vs TV shows |
| 9 | `visualizations/trend_charts/fig_04_additions_over_time.png` | Annual and monthly catalogue additions |
| 10 | `visualizations/trend_charts/fig_05_seasonality_weekday_month.png` | Monthly and weekday seasonality of catalogue additions |
| 11 | `visualizations/trend_charts/fig_06_content_age_gap.png` | Content age at time of upload |
| 12 | `visualizations/correlation_charts/fig_08_correlation_heatmap.png` | Correlation heatmap (movies-only and mixed) |
| 13 | `visualizations/comparison_charts/fig_13_country_segments.png` | Country segmentation: volume and average movie runtime |
| 14 | `visualizations/comparison_charts/fig_14_genre_segments.png` | Genre segmentation by content type |
| 15 | `visualizations/trend_charts/fig_15_audience_mix_over_time.png` | Audience mix over time |

Groups: `distribution_charts/`, `trend_charts/`, `comparison_charts/`, `correlation_charts/`. Each chart has a title, labelled axes with units and annotated values.

## Tools & Technologies

- Python, pandas, NumPy, matplotlib, seaborn, Jupyter.

## Project Structure

```text
EDA-Task-2/
README.md
requirements.txt
data/cleaned_dataset.csv
notebooks/exploratory_data_analysis.ipynb
src/eda_analysis.py
src/build_deliverables.py
visualizations/<group>/fig_*.png
reports/EDA_Report.md + eda_summary.json
```

## How to Run

```bash
pip install -r requirements.txt
python src/eda_analysis.py
python src/build_deliverables.py
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```
