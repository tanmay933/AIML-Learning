# Time Series Cheat Sheet

## Core Concepts (Rapid Recall)

### Components / Concepts

| Concept | One-line meaning | What it impacts | Typical engineering move |
|---|---|---|---|
| Trend | long-term up/down movement | baselines drift, model lag | include trend-capable model (Holt), differencing (ARIMA), time index features |
| Seasonality | repeating pattern with fixed period (e.g., weekly) | predictable oscillations | seasonal naive baseline, Holt-Winters, seasonal features |
| Noise | unpredictable variation | limits achievable accuracy | use robust metrics, avoid overfitting |
| Stationarity | “stable-ish” stats over time | classical model assumptions | differencing / transformations; monitor regime changes |
| Lag | past values used to predict future | core signal source | lag features (t-1, t-7) |
| Rolling stats | past-window aggregates (mean/std) | captures local level/volatility | compute with past-only windows (shift before rolling) |
| Forecast horizon | how far ahead you predict | difficulty + design | choose horizon strategy; longer horizon needs seasonality/exog awareness |

### Quick “what matters most”
- If you see strong weekly repetition: **seasonality dominates**.
- If forecast lags behind real values: **trend handling** is missing.
- If metrics swing wildly: **noise/outliers** or **leakage** is likely.

---

## Classical Models (When to use)

### Moving Average (MA)
- Use when: stable series, simple smoothing baseline.
- Strength: cheap, quick.
- Weakness: lags behind trend; washes out seasonality if window mischosen.

### Exponential Smoothing (ES)
- Use when: practical baseline for business series.
- Strength: reacts to changes better than MA; variants cover trend and seasonality (Holt-Winters).
- Weakness: needs correct seasonal period; can struggle with sharp structural breaks.

### ARIMA (intuition)
- Use when: you need a stronger univariate baseline and can do time-aware validation.
- Strength: handles trend via differencing; models autocorrelation and error correction.
- Weakness: parameter sensitivity; can be expensive at scale; fragile under regime shifts.

---

## 1) Forecasting Workflow (Production-style)

```mermaid
flowchart TD
  A[Define target + horizon + metric] --> B[Collect historical data]
  B --> C[Cleaning + frequency alignment]
  C --> D[Baselines: naive + seasonal naive]
  D --> E[Feature engineering: lags + rolling + calendar]
  E --> F[Train with chronological split / walk-forward]
  F --> G[Forecast]
  G --> H[Evaluate + slice analysis]
  H --> I[Deploy (batch) + cache]
  I --> J[Monitor + retrain cadence]
  J --> F
```

**Non-negotiables**
- Chronological split (no random split)
- Baselines first
- Monitoring and retraining plan from day one

---

## 2) Model Selection Guide (Fast)

### Quick selection table

| Situation | Start with | Next step if needed |
|---|---|---|
| Need immediate baseline | naive + seasonal naive | exponential smoothing |
| Strong seasonality (known period) | seasonal naive | Holt-Winters |
| Mild trend + level changes | exponential smoothing | ARIMA/SARIMAX (if justified) |
| Want stronger univariate classical | Holt-Winters | ARIMA |
| Long horizon with complex drivers | classical baseline first | ML with lag/calendar/exogenous (if available) |

### “When is ARIMA enough?”
ARIMA is often enough when:
- you have a single (or few) important univariate series
- patterns are reasonably stable after differencing/seasonality handling
- you can afford fitting/re-fitting and validate properly

If external drivers dominate (promos/weather) or you have many series at scale, consider ML (covered elsewhere in the handbook) after baselines.

---

## 3) Metric Selection Guide

### Metrics quick table

| Metric | Best when | Main advantage | Main trap |
|---|---|---|---|
| MAE | general ops evaluation | robust + interpretable | doesn’t heavily punish big misses |
| RMSE | big errors are costly | penalizes large misses | dominated by outliers |
| MAPE | scale-free reporting | “% error” is stakeholder-friendly | breaks near zero; misleading with small denominators |

### Rules for choosing
- Choose **MAE** as default for model comparison.
- Choose **RMSE** when large misses are disproportionately harmful.
- Use **MAPE only** if values are safely away from zero and strictly positive.

---

## 4) Rules of Thumb (30+)

1. Never use random split for time series forecasting.
2. Use chronological split; better yet, use walk-forward validation.
3. Always compare to naive and seasonal naive baselines.
4. Seasonal naive is often hard to beat for strongly seasonal data.
5. Choose the forecast horizon first; it drives model choice and evaluation.
6. MAE is a strong default metric for stable comparison.
7. RMSE is appropriate when large errors matter more than small ones.
8. Avoid MAPE with zeros or near-zeros.
9. Backtests should span multiple seasons/regimes, not one convenient month.
10. Compute rolling features with past-only windows (shift before rolling).
11. If your model “predicts too smoothly,” it may be underfitting or missing seasonality.
12. If your forecast lags behind trend, you need trend handling (Holt/differencing/features).
13. If performance looks too good, suspect leakage in features or splits.
14. Align data frequency with the decision cadence (daily vs weekly vs hourly).
15. Explicitly resample; don’t assume timestamps are regular.
16. Handle missing timestamps intentionally; don’t silently drop them.
17. Separate true zeros from missing data; choose a clear policy.
18. Calendar features (day-of-week/month) often provide cheap gains.
19. Recent history is often more predictive than very old history; consider windowing.
20. Structural breaks (launch/pricing changes) can dominate error; document them.
21. Treat outliers carefully: data errors vs real spikes require different handling.
22. Evaluate by slices (holiday vs non-holiday, weekday vs weekend).
23. If you need interpretability and speed, classical methods are strong.
24. Moving averages lag during rapid changes; don’t overtrust them in trending data.
25. Exponential smoothing is a strong practical baseline for many businesses.
26. ARIMA can be “good enough” but can be fragile if you over-tune.
27. Don’t tune ARIMA parameters on the test set.
28. Keep feature computation identical between training and inference.
29. Batch deployment with caching is often sufficient; don’t over-engineer real-time.
30. Monitor data freshness and missingness; forecasting failures are often pipeline failures.
31. Retraining cadence should match drift rate; schedule it explicitly.
32. If seasonal period is wrong (e.g., using 7 when pattern is 14), everything suffers.
33. Use walk-forward to estimate real-world stability across time windows.
34. Longer horizons are harder; set expectations and consider uncertainty.

---

## 5) Common Mistakes

1. Random train/test split (future leakage).
2. Rolling means computed without shifting (leakage).
3. Evaluating on a single holdout period and over-trusting it.
4. Using MAPE on series with zeros/near-zero values.
5. Ignoring seasonality; then being surprised by weekly error spikes.
6. Using the wrong granularity (daily model for weekly decisions).
7. Misaligned timezones / DST issues for hourly data.
8. Treating missing data as zero without confirming business meaning.
9. Over-tuning ARIMA parameters and shipping a fragile model.
10. Not comparing against seasonal naive / exponential smoothing baselines.
11. No monitoring or retraining plan (“set and forget”).
12. Computing features differently in training vs inference (training-serving skew).

---

## 6) Interview Facts (One-liners)

- Time series violates IID; order matters.
- Random split is usually wrong; use chronological or walk-forward.
- Trend = long-term movement; seasonality = fixed repeating pattern.
- Stationarity matters because many classical methods assume stable behavior.
- Lag features (t-1, t-7) are foundational.
- Rolling stats capture local level/volatility; must be past-only.
- Horizon selection changes problem difficulty and model design.
- Seasonal naive is a strong baseline for seasonal data.
- Exponential smoothing is a robust operational baseline.
- ARIMA = AR (lags) + I (differencing) + MA (error correction) intuition.
- MAE is robust; RMSE punishes large errors.
- MAPE can be invalid when actuals can be zero.
- Walk-forward validation best simulates repeated deployment.
- Forecasting is a system: data quality + monitoring matter as much as modeling.

---

## 7) Decision Trees

### A) Model choice decision tree

```mermaid
flowchart TD
  A[Need a forecast] --> B[Always build baselines]
  B --> C{Seasonality present?}
  C -->|Yes| D[Seasonal naive baseline]
  C -->|No| E[Naive / moving average baseline]
  D --> F{Need production baseline with trend+seasonality?}
  E --> F
  F -->|Yes| G[Exponential smoothing (Holt-Winters)]
  F -->|No| H[Ship baseline + monitor]
  G --> I{Need stronger univariate classical?}
  I -->|Yes| J[ARIMA/SARIMAX]
  I -->|No| K[Deploy + monitor + retrain schedule]
```

### B) Metric choice decision tree

```mermaid
flowchart TD
  A[Choose metric] --> B{Do zeros/near-zeros occur?}
  B -->|Yes| C[Avoid MAPE → use MAE/RMSE]
  B -->|No| D{Are big misses disproportionately costly?}
  D -->|Yes| E[RMSE]
  D -->|No| F[MAE (default); MAPE for stakeholder reporting]
```

### C) Validation strategy decision tree

```mermaid
flowchart TD
  A[Validate model] --> B{Time-dependent series?}
  B -->|Yes| C[Chronological split]
  C --> D{Need robust estimate across time?}
  D -->|Yes| E[Walk-forward validation]
  D -->|No| F[Single holdout (last window)]
  B -->|No| G[Standard splits OK (rare for true forecasting)]
```

---

## 8) One-page Revision

### Concepts (memorize)
- Trend: long-term shift
- Seasonality: fixed repeating pattern
- Noise: unpredictable variance
- Stationarity: stable-ish behavior
- Lag: past values as features
- Rolling stats: windowed history context (past-only)
- Horizon: how far ahead; longer = harder

### Baselines (ship first)
- Naive: last value
- Seasonal naive: last season’s value (e.g., last week for daily data)
- Moving average: smoothing baseline

### Classical models (practical)
- Exponential smoothing (Holt-Winters): strong ops baseline for level/trend/seasonality
- ARIMA: stronger univariate baseline; needs careful time-aware validation

### Evaluation
- MAE: robust default
- RMSE: punishes large misses
- MAPE: only if no zeros/near-zeros

### Non-negotiables
- Chronological split; walk-forward preferred
- Past-only features (shift before rolling)
- Compare to seasonal naive
- Monitor data quality + drift; plan retraining cadence