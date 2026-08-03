# End-to-End Advanced Machine Learning Project

---

## 1. Project Overview

**Project:** Credit Card Fraud Detection (Real-Time Scoring + Manual Review)

**Goal:** Detect fraudulent card transactions in near real-time, reduce fraud loss, and keep false positives low to avoid blocking legitimate customers.

### Deliverables (production-style):

- A reproducible training pipeline (data → features → model → evaluation → artifacts)
- A low-latency inference pipeline (features → score → decision)
- Monitoring + drift detection plan
- Retraining strategy and operational runbooks

### Success criteria (business + ML)

Fraud detection is not "maximize accuracy". It's a cost-sensitive, imbalanced, production problem.

| Category | Metric / KPI | Target (example) | Why it matters |
|----------|--------------|------------------|----------------|
| Business | Fraud loss prevented | +X% vs current baseline | Direct ROI |
| Business | Customer friction | Keep declines < Y% | False positives hurt retention |
| Ops | Manual review volume | ≤ capacity | You have a queue constraint |
| ML (offline) | PR-AUC | Improve vs baseline | Better measure for imbalance |
| ML (offline) | Recall at fixed precision | e.g., Recall @ 95% precision | Matches "don't annoy customers" |
| ML (online) | Chargeback rate, approval rate | improve without harming conversion | Real outcome |

---

## 2. Business Problem

### Problem statement

Given a stream of transactions, decide for each transaction:

- Approve immediately
- Decline / block
- Send to manual review

The model outputs a fraud probability `p(fraud)`; the decision policy uses thresholds and operational constraints.

### Constraints (the part most ML projects fail to document)

- **Latency:** decision in < 50–200ms (typical)
- **Availability:** must work during partial outages (fallback)
- **Explainability:** need reason codes for manual review and customer support
- **Delayed labels:** chargebacks arrive days/weeks later → monitoring must handle delayed truth
- **Adversarial environment:** fraud patterns evolve (concept drift is expected)

### Baseline (what you compare against)

- rules engine (velocity checks, blacklists, risky merchant categories)
- simple model (logistic regression)

You should aim to augment, not blindly replace, existing controls.

---

## 3. Understanding the Dataset

### Dataset choice (for learning + prototyping)

Use the common public dataset: **Credit Card Fraud Detection** (Kaggle / European card transactions).

### What it contains (typical):

- **Time:** seconds elapsed from first transaction (proxy timestamp)
- **Amount:** transaction amount
- **V1..V28:** anonymized PCA-like features
- **Class:** label (1 = fraud, 0 = legit)

### What it's missing (real production will have):

- card/account ID
- merchant ID/category
- device/IP/geo
- channel (web/mobile)
- historical aggregates per card/device/merchant
- rule-engine outputs

This matters because many of the most valuable fraud features are **behavioral aggregates** (velocity, recency, novelty).

### Data characteristics

- Severe class imbalance (fraud is rare)
- Non-stationary patterns (fraud shifts; seasonality exists)
- Label delay in real systems (not in Kaggle)

### Label definition (be explicit)

In production, labels might be:

- confirmed chargeback fraud
- confirmed fraudulent dispute
- internal investigation result

Each has different noise and delay profiles. Document this early.

---

## 4. Exploratory Data Analysis (EDA)

### EDA goals in this project:

- Validate assumptions and data quality
- Understand imbalance and leakage risk
- Build a baseline mental model before modeling

### 4.1 Minimal EDA checklist (fraud)

- Class balance: fraud rate overall and over time
- Feature distributions for Amount and key predictors
- Missing values and "impossible values"
- Time-based drift: does fraud rate spike?
- Correlations are less useful here (features anonymized), but still sanity-check

### 4.2 EDA outputs you should produce (even if you don't plot here)

- Fraud rate by time bucket (e.g., per hour/day)
- Amount distribution for fraud vs legit (log scale)
- Feature summary table (mean/std, missingness, outliers)

### 4.3 Engineering note: EDA that prevents leakage

If you do time splits, **do not compute global normalization/stats on the full dataset before splitting.**

---

## 5. Data Cleaning

Fraud datasets often fail due to subtle "data issues" more than model choice.

### 5.1 Cleaning tasks

- Sort by time and validate monotonicity (or handle ties)
- Ensure label is clean and binary
- Handle missing/inf values
- Winsorize/clip extreme values only if they're data errors (not real fraud signals)
- Decide on amount transformation (log1p is common)

### 5.2 Time handling

Even though `Time` is seconds-from-start in the public dataset, treat it as chronological order.

**Rule:** never use random splits here.

---

## 6. Feature Engineering

Even for "advanced ML", fraud detection is still **feature-driven**.

### 6.1 Feature strategy (what we do here)

We'll implement:

- robust numeric preprocessing
- amount transformation
- optional time-based features derived from Time

In production, you'd add entity-based aggregates (per card/device/merchant) and rule outputs.

### 6.2 Practical feature list (prototype)

| Feature | Type | Why it helps | Leakage risk |
|---------|------|--------------|--------------|
| log_amount = log1p(amount) | numeric | stabilizes heavy-tailed distribution | low |
| hour_of_day (from timestamp) | numeric/categorical | captures periodic behavior | low |
| is_weekend | binary | behavior differs on weekends | low |
| rolling fraud rate by merchant | aggregate | strong signal | high if computed using future |
| tx_count_last_1h by card | aggregate | velocity/behavior anomaly | high if window leaks future |

### 6.3 Guardrails against leakage (non-negotiable)

- Entity aggregates must be computed using **only past events** relative to each transaction time.
- For offline training, features must be generated in a way that mirrors production streaming/batch logic.

### 6.4 Example: preprocessing pipeline (sklearn)

A practical approach:

- Keep preprocessing inside a `Pipeline`
- Fit only on training window
- Serialize the whole pipeline

(Prototype code skeleton)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def log1p_amount(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"])
    return X

numeric_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

preprocess = Pipeline(steps=[
    ("log_amount", FunctionTransformer(log1p_amount, feature_names_out="one-to-one")),
    ("select_and_scale", ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ]), numeric_cols)
        ],
        remainder="drop"
    ))
])
```

**Why RobustScaler?**

- Fraud data often has heavy tails/outliers.
- Robust scaling reduces sensitivity to extreme values.

---

## 7. Choosing the Right Algorithm

### Decision: Gradient Boosting (primary) + Logistic Regression (baseline)

**Baseline model:** Logistic Regression

**Primary model:** Gradient Boosted Trees (GBDT), e.g. `HistGradientBoostingClassifier` (sklearn) or LightGBM/CatBoost in production.

### Why GBDT for this project?

| Requirement | Why GBDT fits |
|-------------|---------------|
| Strong on tabular data | Often top performer with minimal feature work |
| Nonlinear interactions | Captures "if-else" patterns well |
| Handles mixed distributions | Works well without perfect normalization |
| Good accuracy/latency tradeoff | Fast inference on CPU |
| Works with class imbalance strategies | sample weights, thresholding |

### Why not start with deep learning?

- Inputs are tabular; boosting is a dominant baseline.
- Deep learning increases training + deployment complexity.
- Fraud is drift-heavy; simpler models are easier to retrain and debug.

### Alternatives and why we're not choosing them first

| Alternative | Why not first-choice (here) |
|-------------|----------------------------|
| Random Forest | strong baseline but often weaker than boosting + less tunable |
| SVM | scaling issues with large datasets; probability calibration often needed |
| Isolation Forest / anomaly detection | useful for cold start, but supervised labels exist and outperform |
| Pure rules | brittle and hard to maintain alone |

---

## 8. Model Training

### 8.1 Correct splitting strategy (chronological)

Fraud is time-dependent. Use a **chronological split**:

- Train on earlier period
- Validate on later period
- Test on final period

(Prototype)

```python
df = df.sort_values("Time").reset_index(drop=True)
X = df.drop(columns=["Class"])
y = df["Class"]

n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
```

### 8.2 Handling imbalance (practical options)

- Use `class_weight="balanced"` (for linear models)
- Use sample weights (cost-sensitive learning)
- Tune decision thresholds on validation
- Evaluate with PR-AUC and recall at fixed precision (not accuracy)

### 8.3 Training pipeline (baseline + primary)

(Prototype)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

lr = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None
    ))
])

gbdt = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42
    ))
])

lr.fit(X_train, y_train)
gbdt.fit(X_train, y_train)
```

**Engineering note:** In production, you'd likely use LightGBM/CatBoost for better performance and categorical handling. The workflow remains the same.

---

## 9. Hyperparameter Tuning

Hyperparameter tuning is not a "leaderboard" exercise; it's controlled iteration under realistic evaluation.

### 9.1 What to tune for GBDT (high impact)

- `max_depth` / `max_leaf_nodes`: controls complexity
- `learning_rate` and `max_iter`: tradeoff between fit quality and speed
- `min_samples_leaf`: helps generalization
- class weighting / sample weights
- calibration and threshold selection (often more impactful than tiny AUC gains)

### 9.2 How to tune without leakage

- Tune using the **validation window**, not test.
- Prefer walk-forward validation when feasible.

(Prototype randomized tuning)

```python
from sklearn.model_selection import ParameterSampler
import numpy as np

param_grid = {
    "model__max_depth": [3, 4, 6, 8],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_iter": [200, 300, 500],
    "model__min_samples_leaf": [20, 50, 100],
}

candidates = list(ParameterSampler(param_grid, n_iter=12, random_state=42))
```

### 9.3 Select by business-shaped metric

Instead of "best PR-AUC", you often want:

- maximize recall subject to precision ≥ target
- or minimize expected cost using a cost matrix

This makes the model selection aligned with deployment.

---

## 10. Model Evaluation

### 10.1 Offline metrics that matter for fraud

| Metric | Why it matters |
|--------|----------------|
| PR-AUC | meaningful under class imbalance |
| Recall at fixed precision | controls customer friction |
| Confusion matrix at chosen threshold | makes tradeoffs visible |
| Cost-based metric | maps to dollars and queue capacity |
| Calibration | probabilities should be usable for policy thresholds |

### 10.2 Evaluate baseline vs primary

(Prototype)

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

def pr_auc(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    return average_precision_score(y, proba)

print("LR PR-AUC:", pr_auc(lr, X_val, y_val))
print("GBDT PR-AUC:", pr_auc(gbdt, X_val, y_val))
```

### 10.3 Choosing thresholds (turn a score into a decision)

Fraud detection is typically a **policy layer** on top of model scores.

Example policy:

- `p < 0.2%` → approve
- `0.2% ≤ p < 1%` → manual review
- `p ≥ 1%` → decline

**How to select thresholds:**

- Use validation set
- Target constraints:
  - max manual review rate
  - minimum precision for declines

(Prototype threshold selection)

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

proba = gbdt.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, proba)

target_precision = 0.95
idx = np.where(precision[:-1] >= target_precision)[0]
best = idx[np.argmax(recall[idx])] if len(idx) else None
chosen_threshold = thresholds[best] if best is not None else 0.5

print("Chosen threshold:", chosen_threshold)
```

### 10.4 Final evaluation on test (only once)

- Freeze model + thresholds
- Evaluate on test window
- Document results and tradeoffs

---

## 11. Model Explainability

Explainability here is about **operational trust**:

- analysts need reason codes
- engineers need debugging signals
- stakeholders need confidence for deployment

### 11.1 What you can realistically explain

- **Global feature importance:** "what signals matter overall?"
- **Local reason codes:** "why was this transaction flagged?"
- **Decision policy:** "how do thresholds map to actions?"

### 11.2 Practical tools

- Permutation importance (model-agnostic)
- SHAP (powerful but heavier)
- Partial dependence (careful with correlated features)

(Prototype: permutation importance)

```python
from sklearn.inspection import permutation_importance

# Use preprocessed features implicitly via pipeline; permutation works on raw X
r = permutation_importance(
    gbdt, X_val, y_val,
    n_repeats=5,
    random_state=42,
    scoring="average_precision"
)
```

### Important limitation in this dataset

The `V*` features are anonymized; you'll get "V14 matters" not "merchant category matters."

In real systems, ensure at least some human-readable features exist for reason codes.

---

## 12. Error Analysis

This is where models become products.

### 12.1 What to examine

- Top false positives (legit but flagged)
- Top false negatives (fraud but missed)
- Error rates across slices:
  - amount buckets
  - time buckets (night vs day)
  - transaction types / channels (in production)
  - new vs returning cards/devices (in production)

### 12.2 Questions that drive improvements

- Are false positives clustered in certain patterns?
  - e.g., travel purchases, high-ticket items, subscription renewals
- Are false negatives "new fraud types"?
  - suggests concept drift or missing features
- Is performance degrading over time?
  - suggests drift and retraining cadence issues

### 12.3 Output of error analysis (engineering artifacts)

- a short "top 5 failure modes" document
- proposed feature additions or policy changes
- a new set of monitoring slices

---

## 13. Business Insights

### Example insights you can produce (even from a prototype)

- A risk threshold policy that reduces fraud loss with manageable manual review volume.
- Identification of high-risk periods (time-based risk).
- Identifying that amount transformation improves stability.
- An operational recommendation: keep a conservative decline threshold and route borderline cases to review.

### How to communicate results

Stakeholders don't want PR-AUC. They want:

- "If we review the top N risky transactions/day, we catch X% of fraud with Y% false positives."
- "Expected monthly savings: $Z (with confidence intervals)."
- "Operational impact: +K manual reviews/day."

---

## 14. Deployment Considerations

This is the minimum viable production plan.

### 14.1 Saving models

Save the entire preprocessing + model pipeline to avoid training/inference skew.

```python
import joblib

joblib.dump(
    {"model": gbdt, "threshold": chosen_threshold, "version": "fraud_gbdt_v1"},
    "fraud_model_v1.joblib"
)
```

**Best practice**

Include: model version, feature schema hash, training data time range, metrics summary.

### 14.2 Inference pipeline (what actually runs)

A realistic online scoring flow:

```mermaid
flowchart LR
  A[Transaction event] --> B[Feature fetch/compute]
  B --> C[Model score p(fraud)]
  C --> D{Policy thresholds}
  D -->|Approve| E[Approve]
  D -->|Review| F[Manual review queue]
  D -->|Decline| G[Decline + reason codes]
```

**Engineering requirements**

- **Idempotency:** avoid double-scoring the same transaction
- **Feature parity:** same transforms as training
- **Timeouts and fallbacks:** if feature store is slow, degrade gracefully

### 14.3 Monitoring (practical, minimal)

You need monitoring even without immediate labels.

**Data quality monitoring**

- missingness spikes
- feature range violations
- schema changes

**Prediction monitoring**

- average score shifts
- % above thresholds
- manual review volume rate
- decline rate

**Delayed label monitoring**

- PR-AUC / recall at fixed precision over rolling windows once chargebacks arrive

**Drift detection**

- feature distribution drift (top features)
- output score distribution drift
- slice monitoring (geo/channel if available)

### 14.4 Retraining strategy

Fraud changes. Plan for retraining from day one.

- **Scheduled retrain:** weekly or monthly (domain-dependent)
- **Trigger retrain:** if drift/performance alerts fire
- **Backtesting:** use time-based splits/walk-forward
- **Safe rollout:** shadow deploy → canary → full deploy
- **Rollback:** keep previous model available for immediate rollback

---

## 15. Lessons Learned

1. Fraud detection is primarily a **decision system**: model + thresholds + ops constraints.
2. The right split is **chronological**; random splits create misleading confidence.
3. Baselines matter: logistic regression + rules is a serious benchmark.
4. GBDT is a strong default for tabular fraud features.
5. Threshold selection and calibration can matter more than small model metric gains.
6. Monitoring without immediate labels is essential; drift is not optional.
7. Explainability isn't just "nice"—it's required for investigation and trust.

---

## 16. Common Mistakes

1. Using accuracy as the primary metric (misleading under imbalance).
2. Random train/test split (temporal leakage).
3. Training on features that won't exist at inference time (training-serving skew).
4. Computing "historical aggregates" using the full dataset (future leakage).
5. Ignoring cost tradeoffs and review capacity in threshold selection.
6. Shipping without monitoring (score distribution + data quality + delayed labels).
7. No fallback when the model/feature store is down.
8. Over-tuning hyperparameters without improving the data/features.
9. Not versioning feature definitions (silently changes model behavior).
10. Deploying a model without a rollback plan.

---

## 17. Interview Questions (~20)

1. Why is accuracy a bad metric for fraud detection?
2. What metrics would you use instead and why? (PR-AUC, recall@precision, cost)
3. How do you choose a decision threshold in a cost-sensitive setting?
4. How do you split data for fraud detection and why?
5. What's the difference between data drift and concept drift in fraud?
6. How would you monitor a fraud model if labels arrive weeks later?
7. Why are gradient-boosted trees a strong default for tabular data?
8. What baseline would you build first and what does "good" look like?
9. How do you avoid leakage when building velocity/aggregate features?
10. What do you deploy: model only or full pipeline? Why?
11. How do you ensure training/inference preprocessing parity?
12. What's your rollback strategy if a new model increases false positives?
13. How do you do error analysis for false positives vs false negatives?
14. How do feedback loops occur in fraud systems (and how to mitigate)?
15. What monitoring slices would you track (geo/channel/device)?
16. What is calibration and when does it matter?
17. How do you handle class imbalance during training?
18. What is a shadow deployment and why use it?
19. How would you combine rules with ML in production?
20. How do you design retraining cadence and triggers?

---

## 18. Project Review Checklist

Use this as a pre-deploy gate.

### Problem + metrics

- ☐ Business objective defined (loss prevention, friction, review capacity)
- ☐ Offline metric aligns with objective (PR-AUC, recall@precision, cost)
- ☐ Threshold policy defined and validated on a validation window

### Data + splits

- ☐ Chronological split used
- ☐ Leakage checks performed (especially aggregates)
- ☐ Feature availability verified for inference

### Modeling

- ☐ Baseline model trained and documented
- ☐ Primary model shows meaningful lift vs baseline
- ☐ Hyperparameter tuning done without using test set

### Evaluation

- ☐ Test set used once at the end
- ☐ Performance measured by slices (time buckets, amount buckets, segments)
- ☐ Confusion matrix at operating threshold documented

### Explainability + debugging

- ☐ Global importance computed
- ☐ Local reason codes available (where possible)
- ☐ Error analysis report created (top failure modes)

### Deployment readiness

- ☐ Full pipeline artifact saved (preprocess + model)
- ☐ Inference latency tested
- ☐ Monitoring dashboards and alerts configured
- ☐ Retraining plan defined
- ☐ Rollback plan tested

---

## 19. Engineering Best Practices

### 19.1 Project structure (practical)

```
data/ (or references to data sources; avoid committing large datasets)
src/
  features/ (feature builders, leakage-safe aggregations)
  models/ (training, tuning, evaluation)
  inference/ (predict wrapper, input validation)
  monitoring/ (metrics emitters, drift checks)
configs/ (YAML/JSON for parameters)
tests/ (unit tests for feature logic and schema)
README.md (how to train, evaluate, run inference)
```

### 19.2 Reproducibility

- Fix random seeds where applicable
- Log:
  - code version (git SHA)
  - training window (time range)
  - feature schema version
  - metrics summary
- Store artifacts with version tags

### 19.3 Data contracts

- Enforce schema and constraints:
  - types, ranges, required fields
- Validate upstream changes early (data quality gates)

### 19.4 Serving discipline

One inference entrypoint:

- validates inputs
- computes/loads features
- calls the pipeline
- applies policy thresholds
- emits logs/metrics

Keep business policy separate from model code when possible (configurable thresholds)

### 19.5 Monitoring as part of the product

**Dashboards:**

- data health
- output distribution
- decision rates (approve/review/decline)
- delayed label performance

**Alerts:**

- data quality break (page)
- threshold rate spike (ticket/page depending on severity)
- performance regression (when labels arrive)

---

## 20. Final Phase Summary

Phase 2 (Advanced Machine Learning) is about moving from "I can train models" to "I can build reliable ML systems".

### What you should now be able to do (integrated view)

**Modeling**

- Choose strong baselines and iterate systematically
- Use the right model families for the data (GBDT for tabular; specialized methods for other modalities)
- Tune hyperparameters responsibly (no test leakage)

**Evaluation**

- Use metrics that reflect the real objective (not vanity metrics)
- Design correct splits (time-aware when needed)
- Perform error analysis and slice-based evaluation

**Feature thinking**

- Build leakage-safe features and understand the lifecycle of features in production
- Understand the difference between manual features, feature extraction, and representation learning (and when to move on)

**Production readiness**

- Package pipelines, not just models
- Plan for monitoring, drift, and retraining from day one
- Deploy safely (shadow/canary/rollback) and design for failures (fallbacks)

### Advanced domains covered

- **Recommendations:** personalization systems and ranking evaluation mindset
- **Time series:** chronological splits, walk-forward validation, forecasting baselines
- **Drift:** how models degrade after deployment and how to detect/respond

### What Phase 3 will build on

Phase 3 (PyTorch) will change **how models learn representations**, but not the engineering discipline:

- rigorous evaluation
- reproducible pipelines
- monitoring and drift management
- production constraints and safe deployment strategies