# EDA Report -- Netflix Movies and TV Shows

> SWYNEX Task 2. From `data/cleaned_dataset.csv` (Task 1, read-only) via `src/eda_analysis.py`. Charts in `visualizations/`. Values: `reports/eda_summary.json`.

## Executive Summary

The catalogue holds 7,786 titles, of which 69.05% are movies. Additions grew to a peak of 2,152 titles in 2019 and then slowed. Content is recent (median release year 2017) and concentrated in mature-rated, US-produced titles. Runtime is stable, correlations are weak, and all flagged outliers are legitimate catalogue entries.

Catalogue: **7,786 titles** (69.05% Movies). Peak **2,152 in 2019** (-6.6% in 2020). Median release 2017; median upload age 1 yr; 29.4% on Fridays. Runtime stable (~99.3 min); correlations weak; IQR extremes kept.

## Dataset Overview

- **7,786 x 14** (109,004 cells, ~4.44 MB).

- Types: 5,376 Movies vs 2,410 TV Shows (ratio 2.23:1). Top rating TV-MA (2,862). Top country United States (2,556). Top genre International Movies.

## Data Quality

- [PASS] show_id is unique: 7,786 unique ids

- [PASS] no duplicate records: 0 duplicate rows

- [PASS] type only Movie / TV Show: Movie (5,376), TV Show (2,410)

- [PASS] duration_unit matches type: movies in Minutes, TV shows in Seasons

- [PASS] duration_value is positive: min 1, max 312

- [PASS] release_year within 1900-2021: 1925-2021

- [PASS] date_added not in the future: latest 2021-01-16

- [PASS] missing values limited to date_added: date_added: 10 (0.13%)

- [REVIEW] no invisible zero-width space (U+200B) in short text: 6 cells: title 5, cast 1

## Statistical Summary

- Movies: mean 99.3 min, median 98, IQR 86-114, skew 0.191.

- TV: mean 1.78 seasons, 66.7% single-season.

- Release year median 2017; 70.0% from 2015+.

## Key Trends

- 2 (2008) -> 2,152 (2019), then 2,009 in 2020 (-6.6%).

| Year | Titles added | YoY growth |
|---|---|---|
| 2015 | 88 | 252.0% |
| 2016 | 443 | 403.4% |
| 2017 | 1,225 | 176.5% |
| 2018 | 1,685 | 37.6% |
| 2019 | 2,152 | 27.7% |
| 2020 | 2,009 | -6.6% |

- December busiest (833); Fridays 2,286 (29.4%); peak day 2020-01-01 (119) vs mean 5.14.

- Freshness: median age 1 yr; 55.6% within one year.

## Important Relationships

- Runtime vs year r=-0.205; heatmap: 1 strong / 2 moderate / 7 weak. Correlation is not causation.

## Anomalies

- 337 long films, 236 long series, 745 pre-2005 titles. All IQR-flagged observations are legitimate catalogue entries (long films, multi-season series, vintage titles). Nothing was removed.

## Segmentation

- US 2,883 (39.6%); India 956; longest India (126.9 min).

- Genres: 42 distinct; top International Movies (2,436).

## KPIs

| KPI | Value | What it means |
|---|---|---|
| Catalogue size | 7,786 titles | Total rows in cleaned dataset |
| Movie share | 69.05% (5,376) | Films dominate |
| TV Show count | 2,410 | Series are the minority |
| Peak additions year | 2019 (2,152) | Biggest year |
| Growth CAGR 2015-2020 | 86.9% | Expansion then plateau |
| Avg. movie runtime | 99.3 min (median 98) | Feature length |
| Avg. TV length | 1.78 seasons (66.72% single) | Thin depth |
| Median upload age | 1 year(s) | Mostly fresh |
| Fresh content (<=1yr) | 55.58% | Day-and-date share |
| Friday launch share | 29.4% | Weekly rhythm |
| Leading country | United States (32.83%) | Top supplier |
| Leading rating | TV-MA (36.76%) | Adult-skewed |
| Leading genre | International Movies (14.27% tags) | Top token |
| Avg. genres/title | 2.19 | Multi-genre labelling |
| date_added completeness | 99.87% | Temporal coverage |

## Key Insights

### Insight 1 -- Catalogue growth peaked in 2019 and has slowed since

**Finding:** Annual additions ramped up from 2 titles in 2008 to a peak of 2,152 titles in 2019, a 1076.0x increase. Additions then fell 6.6% to 2,009 in 2020.

**Evidence:** 2019 = 2,152 titles; 2020 = 2,009 (-6.6%); 2017 year-over-year growth was 176.5%. The 2015-2020 compound annual growth rate of additions is 86.9%.

**Why it matters:** The platform moved through a rapid library-expansion phase and then slowed the pace of additions. Any forward-looking view of catalogue size should model a plateau rather than continued exponential growth.

**Visualization:** `visualizations/trend_charts/fig_04_additions_over_time.png`

### Insight 2 -- Movies are the bulk of the catalogue, but TV shows are thin in depth

**Finding:** 5,376 of 7,786 titles are Movies (69.05%) versus 2,410 TV Shows. Yet 66.7% of TV shows run for a single season, so the catalogue is wide in films and shallow in long-running series.

**Evidence:** Movie:TV ratio is 2.23:1. Median TV show length is 1 season while the median movie runs 98 minutes.

**Why it matters:** Films deliver instant breadth with low viewing commitment, while the small share of multi-season shows limits binge-depth content. Portfolio decisions can therefore be framed as breadth (films) versus depth (returning series).

**Visualization:** `visualizations/comparison_charts/fig_10_type_composition.png`

### Insight 3 -- Movie runtime is stable and does not track release year

**Finding:** Movie runtime is tightly concentrated (mean 99.3 min, median 98 min, interquartile range roughly 86-114 min) and its correlation with release year is -0.205.

**Evidence:** Across 7 decades of well-populated data, decade means stay inside a 53-minute band, and the distribution is close to symmetric (skew 0.191).

**Why it matters:** Audience expectations of a standard feature length have not shifted, so scheduling, runtime-based recommendations and slot planning can rely on a stable target window regardless of a film's era.

**Visualization:** `visualizations/correlation_charts/fig_07_runtime_vs_release_year.png`

### Insight 4 -- Releases follow a deliberate weekly rhythm: Friday dominates

**Finding:** 2,286 titles (29.4% of all dated additions) were added on a Friday, making it the platform's primary publishing day. December is the busiest month with 833 titles.

**Evidence:** Mean additions per calendar day is only 5.14, yet the single busiest day (2020-01-01) carried 119 titles - a clear batch-release pattern.

**Why it matters:** A Friday-first cadence concentrates attention and marketing spend into one weekly window. Competitor planning and any analysis of weekend viewing spikes must account for this deliberate scheduling.

**Visualization:** `visualizations/trend_charts/fig_05_seasonality_weekday_month.png`

### Insight 5 -- The library has shifted towards fresh, near-current content

**Finding:** The median title is added 1 year after its original release (mean 4.6 years), and 55.6% of dated titles were added within one year of release.

**Evidence:** 70.0% of the catalogue was released in 2015 or later and the median release year is 2017, while the oldest title dates to 1925.

**Why it matters:** A catalogue concentrated on recent material competes on newness rather than on a deep archive, which raises the importance of continuous licensing and original production.

**Visualization:** `visualizations/trend_charts/fig_06_content_age_gap.png`

### Insight 6 -- Content supply is highly concentrated in a handful of countries

**Finding:** 39.6% of titles come from the United States and the top five countries supply 67.5% of the catalogue; India alone contributes 956 titles.

**Evidence:** 679 distinct countries appear in the data. Among the 23 countries with at least 30 movies, India has the longest average runtime (126.9 min) against a global movie mean of 99.3 min, while Canada averages 86.7 min.

**Why it matters:** Sourcing is geographically concentrated, which is both a supply-chain risk and an opportunity: strong producers that are currently under-represented are candidate markets for localised content investment.

**Visualization:** `visualizations/comparison_charts/fig_13_country_segments.png`

### Insight 7 -- Mature-rated content anchors the catalogue and its share is stable over time

**Finding:** TV-MA alone accounts for 36.76% of titles, and the mature audience segment represents 71.3% of the catalogue.

**Evidence:** 70.1% of titles carry a mature rating versus 13.3% family/kids. The mature share stood at 59.09% in 2015 and 61.54% in the final year, so the mix is not shifting with catalogue growth.

**Why it matters:** Advertising, compliance and parental-control features must be sized for an adult-majority library, while the family segment stays a meaningful but secondary audience.

**Visualization:** `visualizations/comparison_charts/fig_09_rating_distribution.png`

### Insight 8 -- Statistical outliers are legitimate long-tail content, not data errors

**Finding:** IQR screening flags 337 movies (4.3%), 236 TV shows (3.0%) and 745 pre-2005 titles.

**Evidence:** Flagged cases include feature films of up to 312 minutes and series of up to 16 seasons, with the oldest title dated 1925.

**Why it matters:** These extremes serve genuine long-format and vintage audiences. They should be segmented and reported separately rather than trimmed, because deleting them would bias average runtime and hide the long tail.

**Visualization:** `visualizations/distribution_charts/fig_02_duration_boxplot.png`

### Insight 9 -- Numeric fields carry very little genuine correlation

**Finding:** Of 10 movie-level field pairs, 1 are strong (|r| >= 0.5) and 2 moderate; the remainder are weak.

**Evidence:** The strongest relationship is release year versus content age at upload (r = -0.99), which is a calendar artefact rather than a behavioural driver. Runtime versus release year is r = -0.205.

**Why it matters:** Catalogue metadata alone cannot explain performance because the dataset contains no viewership, revenue or engagement measures. Correlation must not be read as causation here, and any predictive modelling needs additional behavioural data.

**Visualization:** `visualizations/correlation_charts/fig_08_correlation_heatmap.png`

### Insight 10 -- Metadata completeness differs by field and is reported honestly

**Finding:** Missing metadata is concentrated in credits: director is unspecified for 30.68% of titles and the placeholder country label covers 507 titles. Only date_added is genuinely empty (10 cells, 0.01% of the dataset).

**Evidence:** date_added completeness is 99.87%. A residual formatting issue was also found: 6 text cells still contain an invisible zero-width space that Task 1's whitespace rule (str.strip) cannot detect.

**Why it matters:** Crew-level and country-level analysis will under-count because of unspecified credits, so those results are stated as shares of titles with known credits. The zero-width space residue is a minor but concrete improvement for Task 1's pipeline.

**Visualization:** `visualizations/distribution_charts/fig_03_release_year_distribution.png`

## Visualizations

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

## Conclusion

The catalogue holds 7,786 titles, of which 69.05% are movies. Additions grew to a peak of 2,152 titles in 2019 and then slowed. Content is recent (median release year 2017) and concentrated in mature-rated, US-produced titles. Runtime is stable, correlations are weak, and all flagged outliers are legitimate catalogue entries.

## Limitations

- Snapshot ends 2021-01-16; no viewership/revenue columns.

- Country/crew placeholders under-count; r=-0.99 is calendar artefact.

- IQR flags are descriptive, not error proof.
