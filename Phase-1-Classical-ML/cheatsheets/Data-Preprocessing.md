# Data Preprocessing Checklist

## The Preprocessing Pipeline (Overview)

```
Raw Data
   ↓
Check & Handle Missing Values
   ↓
Remove Duplicates
   ↓
Detect & Treat Outliers
   ↓
Split into Train/Test (BEFORE fitting any transformer)
   ↓
Feature Engineering
   ↓
Encode Categorical Variables (fit on train only)
   ↓
Scale/Normalize Numerical Features (fit on train only)
   ↓
Feature Selection
   ↓
Verify No Data Leakage
   ↓
Train Model
```

**⭐ Golden Rule**: Any transformation that *learns* something from the data (mean, std, min/max, categories, feature relevance) must be fit **only on training data**, then applied to test data — never the reverse.

---

## 1. Missing Values

| Aspect | Detail |
|---|---|
| **Why?** | Most algorithms can't process `NaN`/`null` directly, and silently ignoring them can produce distorted or broken results |
| **When?** | Always check first, right after loading data — before anything else |
| **How?** | Drop rows/columns, or impute with Mean/Median/Mode depending on data type and distribution |

| Strategy | Use When |
|---|---|
| Drop rows | Missingness is rare (<5%) and random |
| Drop columns | Feature missing in majority of rows |
| Mean Imputation | Numeric, symmetric distribution, no major outliers |
| Median Imputation | Numeric, skewed distribution or outliers present |
| Mode Imputation | Categorical data |

**⚠ Common Mistakes**
- Ignoring missing values and letting the model/library silently handle them incorrectly
- Using mean imputation on heavily skewed data (distorted by outliers)
- Imputing before understanding *why* data is missing (missingness itself can be informative)
- Fitting the imputer on the full dataset instead of training data only

---

## 2. Duplicate Data

| Aspect | Detail |
|---|---|
| **Why?** | Duplicate rows give the model artificially more "votes" for the same data point, biasing training and inflating evaluation if duplicates span train/test |
| **When?** | Right after checking missing values, during initial data cleaning |
| **How?** | Check exact duplicates across all columns, or key-column duplicates (e.g., same `user_id` + `timestamp`); drop keeping first occurrence |

**⚠ Common Mistakes**
- Only checking for exact duplicates, missing near-duplicates from formatting inconsistencies (e.g., `"NY"` vs `"New York"`)
- Not checking for duplicates that span the train/test boundary — a subtle leakage source

---

## 3. Outliers

| Aspect | Detail |
|---|---|
| **Why?** | Extreme values can distort statistics (mean, variance), distort model training (especially squared-error-based models), and mislead distance-based algorithms |
| **When?** | After handling missing values/duplicates, before scaling |
| **How?** | Visualize distributions, use IQR method or Z-score, then decide with **domain knowledge** whether to remove, cap, or keep |

```
IQR Method:
     Q1            Median            Q3
      |----------------|----------------|
      |<-- middle 50% of the data -->|
Outliers: far below Q1 or far above Q3
```

| Outlier Type | Action |
|---|---|
| Data entry error / sensor glitch | Remove or correct |
| Genuine rare event (relevant to the task) | Keep |
| Anomaly detection target (e.g., fraud) | Definitely keep — it's the signal |

**⚠ Common Mistakes**
- Blindly removing all statistical outliers without checking if they're real, meaningful data points
- Removing outliers that are actually the entire point of the model (e.g., fraud detection)
- Detecting outliers using the full dataset instead of training data only

---

## 4. Feature Scaling (Standardization & Normalization)

| Aspect | Detail |
|---|---|
| **Why?** | Distance-based and gradient-based algorithms are sensitive to feature magnitude — unscaled features with larger ranges dominate calculations unfairly |
| **When?** | After outlier treatment, right before (or as part of) model training — always fit after the train/test split |
| **How?** | Standardize (mean 0, std 1) or Normalize (fixed range, typically [0,1]) using training data statistics only |

| Technique | What It Does | Outlier Sensitivity | Best For |
|---|---|---|---|
| **Standardization** | Centers to mean 0, std 1 | Less sensitive | Default choice, most algorithms |
| **Normalization (Min-Max)** | Rescales to a fixed range [0,1] | Very sensitive | Bounded ranges needed, few outliers |

### Which Algorithms Need Scaling?

| Needs Scaling | Doesn't Need Scaling |
|---|---|
| KNN | Decision Trees |
| Linear/Logistic Regression | Random Forest |
| KMeans | Boosting (tree-based) |
| PCA | — |

**⚠ Common Mistakes**
- Scaling before the train/test split (data leakage)
- Assuming tree-based models need scaling (they don't)
- Using Normalization on data with significant outliers (compresses everything else into a tiny range)
- Forgetting to apply the *same* fitted scaler to the test set (never re-fit on test data)

---

## 5. Encoding Categorical Variables

| Aspect | Detail |
|---|---|
| **Why?** | Models only understand numbers — categorical text must be converted before training |
| **When?** | After splitting, fit only on training data |
| **How?** | Choose Label Encoding (ordinal) or One-Hot Encoding (nominal) based on whether the category has a meaningful order |

| Encoding | Use For | Risk |
|---|---|---|
| **Label Encoding** | Ordinal data (has natural order) | Implies false order if used on nominal data |
| **One-Hot Encoding** | Nominal data (no order), low cardinality | Increases dimensionality with many categories |

```
Decision:
  Does the category have a natural order? (e.g., Low < Medium < High)
        │
        ├── YES → Label Encoding (preserve the order explicitly)
        └── NO  → One-Hot Encoding
                    │
                    └── Too many unique categories (high cardinality)?
                          → Consider grouping rare categories into "Other"
```

**⚠ Common Mistakes**
- Using Label Encoding on nominal data (e.g., City names) — introduces a fake numeric relationship
- One-Hot Encoding a high-cardinality feature without first grouping rare categories
- Fitting the encoder on the full dataset (including test data) before splitting

---

## 6. Feature Engineering

| Aspect | Detail |
|---|---|
| **Why?** | Well-engineered features often improve model performance more than switching algorithms — they expose meaningful patterns directly |
| **When?** | After understanding the data (EDA), can happen before or after splitting depending on whether it uses only single-row info or aggregate statistics |
| **How?** | Extract components from existing features (e.g., Date → Day of Week) or create new combined features (e.g., Income/Expenses → Savings Rate) |

| Type | Example |
|---|---|
| **Feature Extraction** | `Date` → `Day of Week` |
| **Feature Creation** | `Income`, `Expenses` → `Savings Rate` |

**⚠ Common Mistakes**
- Skipping feature engineering entirely and relying only on raw features
- Creating aggregate features (e.g., "average purchase per customer") using data that spans across the train/test split, causing leakage
- Creating features using information not available at real prediction time (Section 9)

---

## 7. Feature Selection

| Aspect | Detail |
|---|---|
| **Why?** | Removing irrelevant/redundant features reduces overfitting risk, speeds up training, and improves interpretability |
| **When?** | After feature engineering, before final model training |
| **How?** | Use correlation analysis, feature importance (from tree-based models), or regularization (L1/Lasso naturally zeroes out weak features) |

| Method | Approach |
|---|---|
| Correlation Matrix | Drop one of two highly correlated features (multicollinearity) |
| Tree-based Feature Importance | Keep only top-N most important features |
| L1 (Lasso) Regularization | Automatically zeroes out irrelevant feature coefficients |

**⚠ Common Mistakes**
- Selecting features by looking at the entire dataset, including test data
- Removing features solely based on low correlation with the target, ignoring potential non-linear relevance
- Conflating Feature Selection (keeping a subset of original features) with PCA (creating entirely new, blended features)

---

## 8. Data Leakage

| Aspect | Detail |
|---|---|
| **Why?** | Leakage causes a model to look artificially good during development, then fail in real-world production use |
| **When?** | A risk at EVERY preprocessing step — must be checked continuously, not just once |
| **How?** | Always split first; fit every transformer (scaler, encoder, imputer, PCA, feature selector) only on training data |

| Leakage Source | Example |
|---|---|
| Scaling before splitting | Scaler statistics computed using test data too |
| Using future information | A feature that wouldn't exist at prediction time |
| Target leakage | A feature that's essentially a proxy for the answer |
| Resampling before splitting | SMOTE/oversampling applied to the full dataset |
| Duplicate rows across train/test | Same data point appears in both sets |

```
CORRECT ORDER:
Raw Data → Split (Train/Test) → Fit transformer on TRAIN only
              → Apply (transform, not re-fit) to TEST
```

**⚠ Common Mistakes**
- Fitting ANY transformer (scaler, encoder, imputer, PCA) on the full dataset before splitting
- Engineering aggregate features using statistics that span both train and test data
- Including a feature that directly or indirectly encodes the target

---

## 9. Train/Test Split

| Aspect | Detail |
|---|---|
| **Why?** | Need to evaluate on genuinely unseen data to estimate real-world generalization |
| **When?** | As early as possible — ideally right after initial cleaning (missing values, duplicates), before any transformer is fit |
| **How?** | Random split (regression) or **stratified** split (classification, especially imbalanced) — typical ratios 80/20 or 70/15/15 |

| Split Type | Use When |
|---|---|
| Random Split | Regression, or balanced classification |
| Stratified Split | Classification with class imbalance — preserves class proportions |

**⚠ Common Mistakes**
- Splitting after fitting scalers/encoders (leakage)
- Not using a stratified split on imbalanced classification data
- Using random splits on time-series data (should respect chronological order instead)
- Resampling (e.g., SMOTE) the test set — it must reflect the real-world distribution

---

## Preprocessing Decision Flowchart

```
START
  │
  ▼
Check for missing values → impute or drop (Section 1)
  │
  ▼
Check for duplicates → remove (Section 2)
  │
  ▼
Check for outliers → investigate with domain knowledge (Section 3)
  │
  ▼
SPLIT INTO TRAIN/TEST (stratify if classification + imbalanced)
  │
  ▼
Feature Engineering (single-row features safe either side; aggregate
features must respect the split)
  │
  ▼
Fit Encoder on TRAIN only → transform TRAIN and TEST (Section 5)
  │
  ▼
Does the chosen algorithm need scaling?
  │
  ├── YES (KNN, Linear/Logistic Regression, KMeans, PCA)
  │        → Fit Scaler on TRAIN only → transform TRAIN and TEST
  │
  └── NO (Decision Trees, Random Forest, Boosting)
           → Skip scaling
  │
  ▼
Feature Selection (based on TRAIN data only) (Section 7)
  │
  ▼
Double-check for Data Leakage (Section 8)
  │
  ▼
TRAIN MODEL
```

---

## Preprocessing Comparison Tables

### Missing Value Strategy Quick Reference

| Data Missing | Data Type | Recommended Strategy |
|---|---|---|
| Small %, random | Numeric, symmetric | Mean Imputation |
| Small %, random | Numeric, skewed/outliers | Median Imputation |
| Small %, random | Categorical | Mode Imputation |
| Majority of column | Any | Drop column |

### Scaling Quick Reference

| Algorithm | Scaling Required? |
|---|---|
| Linear/Logistic Regression | Recommended |
| KNN | Mandatory |
| KMeans | Mandatory |
| PCA | Mandatory |
| Decision Trees | No |
| Random Forest | No |
| Boosting | No |

### Encoding Quick Reference

| Data Type | Encoding |
|---|---|
| Ordinal (ordered categories) | Label Encoding |
| Nominal, low cardinality | One-Hot Encoding |
| Nominal, high cardinality | Group rare categories first, then encode |

---

## Interview Notes

🎯 **"Why is data leakage dangerous?"**
> It produces performance estimates that look great during development but don't hold up in production, because the model was trained (directly or indirectly) using information it wouldn't actually have at real prediction time.

🎯 **"Do all algorithms need feature scaling?"**
> No — distance-based (KNN, KMeans) and gradient/variance-based (Linear/Logistic Regression, PCA) algorithms need it. Tree-based algorithms (Decision Trees, Random Forest, Boosting) split on thresholds and are unaffected by scale.

🎯 **"Mean vs Median imputation — how do you decide?"**
> Use median when the data is skewed or has outliers, since mean is pulled toward extreme values. Use mean when the distribution is roughly symmetric.

🎯 **"One-Hot vs Label Encoding — how do you decide?"**
> If the category has a meaningful order (ordinal), use Label Encoding. If not (nominal), use One-Hot Encoding to avoid implying a false order.

🎯 **"When should outliers be removed vs kept?"**
> Remove them if they're data entry errors or measurement glitches. Keep them if they represent genuine rare events relevant to the problem — especially if the outliers themselves are the prediction target (e.g., fraud, anomalies).

---

# Before Training Any Model Checklist

- [ ] Checked for and handled missing values (with a justified strategy, not a default)
- [ ] Removed exact and near-duplicate records
- [ ] Investigated outliers using domain knowledge — decided keep/remove/cap deliberately
- [ ] Split data into Train/Test (stratified if classification + imbalanced) **before** fitting any transformer
- [ ] Performed feature engineering, being careful with aggregate features that could leak across the split
- [ ] Encoded categorical variables correctly (Label for ordinal, One-Hot for nominal), fit on train only
- [ ] Determined whether the chosen algorithm requires scaling — scaled if needed, fit on train only
- [ ] Performed feature selection using training data only
- [ ] Verified no data leakage exists anywhere in the pipeline (no future info, no target proxies, no full-dataset fitting)
- [ ] Confirmed the test set was never touched by any resampling, scaling, or encoding fit
- [ ] Confirmed the pipeline is reproducible: the exact same transformations applied to train are applied to test/inference data
- [ ] Ready to train — all preprocessing decisions are documented and justified