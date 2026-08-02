# Machine Learning Revision Sheet

---

## 1. ML Fundamentals

```
AI ⊃ ML ⊃ DL
```

| Term | Meaning |
|---|---|
| AI | Any system mimicking intelligent behavior (rules or learning) |
| ML | Learns patterns from data instead of hardcoded rules |
| DL | ML using multi-layer neural networks |

| Supervised | Unsupervised | Reinforcement |
|---|---|---|
| Labeled data (X→y) | Unlabeled, find structure | Agent/Environment/Reward |
| Regression / Classification | Clustering / Dim. Reduction | Trial-and-error learning |

**⭐ Must Remember**: X = features, y = target. Train ≠ Test (unseen data = generalization). Model = Algorithm + learned Parameters. Hyperparameter = set before training; Parameter = learned during training.

```
Workflow: Problem → Collect → Understand → Prepare → Split → Choose Model
          → Train → Validate → Final Eval → Deploy → Monitor
```

---

## 2. Data Preparation

| Missing Values | Best For |
|---|---|
| Drop rows/cols | Small % missing / feature mostly missing |
| Mean | Symmetric, no outliers |
| Median | Skewed / outliers present |
| Mode | Categorical |

| Scaling | Outlier Sensitive? | Range |
|---|---|---|
| Standardization | Less | mean 0, std 1 |
| Normalization (Min-Max) | Very | [0,1] |

| Encoding | Use For |
|---|---|
| Label Encoding | Ordinal (order matters) |
| One-Hot Encoding | Nominal (no order) |

**⭐ Must Remember**: Split BEFORE scaling/encoding — fitting on full data = **Data Leakage**. Trees don't need scaling; KNN/Linear/Logistic Regression do.

```
Correct order: Raw Data → Split → Fit scaler on TRAIN only → Transform both
```

---

## 3. Linear Regression

```
y = w1x1 + w2x2 + ... + wn xn + b
```

- Coefficient = effect of feature holding others constant
- Cost Function: MSE (minimized via Gradient Descent)
- Fit line minimizes squared residuals

**🎯 Interview Fact**: Sensitive to outliers (squared error). Assumes linearity. Fast, interpretable baseline. Bad for stock prediction (non-linear, noisy).

---

## 4. Regression Metrics

| Metric | Formula | Notes |
|---|---|---|
| MAE | avg\|actual−pred\| | Robust to outliers |
| MSE | avg(actual−pred)² | Penalizes large errors heavily |
| RMSE | √MSE | Same units as target |
| R² | 1 − (model err/baseline err) | Can be negative |
| Adjusted R² | R² penalized for extra features | Use when comparing feature counts |

**⭐ Must Remember**: RMSE ≥ MAE always. Big gap between them = outliers dominating. R²=0 → same as predicting mean. R² always ↑ with more features; Adjusted R² doesn't.

---

## 5. Linear Regression Assumptions

| Assumption | Detected By | If Violated |
|---|---|---|
| Linearity | Residual vs Predicted (curve) | Underfitting |
| Independence | Residual vs Time (pattern) | Unreliable coefficients (time series) |
| Homoscedasticity | Residual vs Predicted (funnel) | Inconsistent accuracy |
| Normality of Residuals | Histogram/Q-Q plot | Bad confidence intervals |
| No Multicollinearity | Correlation matrix / VIF | Unstable coefficients |

**🎯 Interview Fact**: Correlation ≠ Causation (ice cream/drowning example). High R² does NOT confirm assumptions hold.

---

## 6. Polynomial Regression

```
y = w1x + w2x² + w3x³ + b   (still Linear Regression, just engineered features)
```

| Degree | Effect |
|---|---|
| Low | Underfitting |
| Just right | Good fit |
| High | Overfitting (wiggly curve) |

**⚠ Must Remember**: Choose degree using TEST error, not training error. Poor extrapolation outside training range. Needs feature scaling (x³ >> x).

---

## 7. Bias vs Variance

| | Bias | Variance |
|---|---|---|
| Cause | Model too simple | Model too sensitive to training data |
| Symptom | Underfitting | Overfitting |
| Train/Test Error | Both high, similar | Train low, Test high (gap) |

```
Complexity ↑ → Bias ↓, Variance ↑
```

**Fixes**: High Bias → more complexity, more features, less regularization.
High Variance → simplify model, more data, regularization, ensembles.

**🎯 Interview Fact**: Total Error = Bias² + Variance + Irreducible Noise. Applies to ALL models, not just regression.

---

## 8. Regularization

```
Regularized Cost = Original Cost + λ × Penalty(weights)
```

| | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalty | \|w\| | w² |
| Feature Selection | Yes (zeros out) | No |
| Multicollinearity | Arbitrary pick | Stable |

**Elastic Net** = L1 + L2 combined.

**⭐ Must Remember**: λ=0 → no regularization. λ too high → underfitting. Requires feature scaling.

---

## 9. Cross Validation

```
K-Fold: split data into K folds → rotate test fold → average scores
```

| Type | Use Case |
|---|---|
| K-Fold | General purpose (K=5 or 10 standard) |
| Stratified K-Fold | Classification, imbalanced classes |
| LOOCV | Very small datasets (K = N) |
| Nested CV | Tuning + unbiased final estimate together |

**⚠ Must Remember**: Preprocessing must be fit PER FOLD, not on full data. CV complements, doesn't replace, a final held-out test set.

---

## 10. Logistic Regression

```
z = w1x1 + ... + b
P(Class 1) = Sigmoid(z) = 1 / (1 + e^-z)
```

- Models **log-odds** as linear function
- Decision boundary default: P ≥ 0.5 → Class 1
- Cost Function: **Log Loss** (not MSE)

**🎯 Interview Fact**: Coefficients affect log-odds, not outcome directly. `predict_proba()` > `predict()` when threshold needs tuning.

---

## 11. Classification Metrics

```
              Predicted +      Predicted −
Actual +      TP                FN
Actual −      FP                TN
```

| Metric | Formula | Answers |
|---|---|---|
| Accuracy | (TP+TN)/Total | Overall correctness |
| Precision | TP/(TP+FP) | Trustworthy positives? |
| Recall | TP/(TP+FN) | Caught all positives? |
| F1 | 2PR/(P+R) | Balance of both |
| Specificity | TN/(TN+FP) | Caught all negatives? |
| FPR | FP/(FP+TN) = 1−Specificity | False alarm rate |

**⭐ Must Remember**: Accuracy misleading on imbalanced data. ROC-AUC = ranking ability across thresholds. PR-Curve preferred over ROC on imbalanced data.

---

## 12. Imbalanced Data

| Technique | Changes | Risk |
|---|---|---|
| Under-sampling | Removes majority rows | Loses data |
| Over-sampling | Duplicates minority rows | Overfitting |
| SMOTE | Synthetic minority rows | Needs enough minority samples |
| Class Weights | Cost function penalty | Needs algorithm support |
| Threshold Adjustment | Decision cutoff | Cheapest fix, no retrain |

**⚠ Must Remember**: NEVER resample the test set. Never trust accuracy — use Precision/Recall/F1/PR-AUC.

---

## 13. KNN

```
Prediction = majority vote (classification) / average (regression) of K nearest neighbors
```

| K | Effect |
|---|---|
| Small K | High Variance (overfitting) |
| Large K | High Bias (underfitting) |

**⭐ Must Remember**: Lazy learner (no training). Requires feature scaling (distance-based). Curse of Dimensionality hurts it in high-D. Training O(1), Prediction O(n×d).

---

## 14. Decision Trees

```
Recursive Splitting: pick split with highest Information Gain → repeat → stop at leaf
```

| Metric | Range | Notes |
|---|---|---|
| Gini Impurity | 0–0.5 | Faster (default) |
| Entropy | 0–1 | Log-based |

```
Information Gain = Impurity(before) − Weighted Impurity(after)
```

**⭐ Must Remember**: No scaling needed. Overfits without max_depth/min_samples_leaf. Unstable (high variance) — motivates Random Forest. Prediction O(log n) — very fast.

---

## 15. Random Forest

```
Random Forest = Bagging (bootstrap rows) + Random Feature Selection (per split)
```

- Trees trained **independently, in parallel**
- Classification → majority vote; Regression → average
- **OOB Error**: free validation using ~37% left-out rows per tree

**🎯 Interview Fact**: More trees (n_estimators) does NOT increase overfitting — it stabilizes. Reduces Variance (Ch.7), not Bias. No scaling needed.

---

## 16. Boosting

```
Sequential weak learners: each new tree corrects previous errors
```

| | AdaBoost | Gradient Boosting |
|---|---|---|
| Mechanism | Reweight misclassified points | Fit residual errors |

| | Random Forest | Boosting |
|---|---|---|
| Training | Parallel | Sequential |
| Fixes | Variance | Bias |
| More trees/rounds | Safe | Can overfit |

XGBoost/LightGBM/CatBoost = optimized Gradient Boosting variants.

**⚠ Must Remember**: learning_rate ↓ needs n_estimators ↑. Cannot parallelize across trees (sequential dependency).

---

## 17. KMeans

```
1. Init K centroids (K-Means++) 
2. Assign points to nearest centroid
3. Update centroids = avg of assigned points
4. Repeat until convergence
```

| Method | Use |
|---|---|
| Elbow Method | Plot inertia vs K, find "elbow" |
| Silhouette Score | Accounts for separation too (-1 to 1) |

**⭐ Must Remember**: Unsupervised (no y). Requires feature scaling. Assumes spherical, similar-sized clusters. Sensitive to init & outliers. K-Means++ = smart init (sklearn default).

---

## 18. PCA

```
Finds orthogonal directions (Principal Components) of MAXIMUM VARIANCE
```

- Unsupervised — never looks at y
- Requires feature scaling (mandatory)
- Explained Variance Ratio → decide # components (target 90-95% cumulative)

| | Feature Selection | PCA |
|---|---|---|
| Output | Subset of original features | New blended features |
| Interpretability | High | Low |
| Uses y? | Often | Never |

**⚠ Must Remember**: Fights Curse of Dimensionality. Fit only on train, transform test (leakage rule). Can discard task-relevant low-variance info.

---

## 19. End-to-End ML Workflow

```
Problem Definition → Collect Data → EDA → Clean → Feature Engineer
   → Encode/Scale → Split (stratified if imbalanced) → Baseline Model
   → Cross Validate → Tune Hyperparameters → Select Model
   → Error Analysis → Save Model → Deploy → Monitor → Watch Drift
```

**⭐ Must Remember**: Define metric BEFORE modeling. Always try simple baseline first. Error analysis often loops back to feature engineering. Model drift = real-world data shifts, performance degrades → retrain.

---

# Last Minute Interview Revision Checklist

- [ ] Can define AI vs ML vs DL, and Supervised vs Unsupervised vs RL
- [ ] Can explain train/test split, generalization, and overfitting in one breath
- [ ] Know when to use Mean vs Median imputation, Label vs One-Hot encoding
- [ ] Can state Data Leakage rule: split BEFORE fit any transformer
- [ ] Can write Linear Regression equation and explain coefficients
- [ ] Know MAE vs MSE vs RMSE vs R² vs Adjusted R² — formulas + when to use each
- [ ] Can list all 5 Linear Regression assumptions and how each is checked
- [ ] Can explain why Polynomial Regression is "still Linear Regression"
- [ ] Can draw the Bias-Variance tradeoff curve and diagnose from train/test error gap
- [ ] Can explain L1 vs L2 Regularization and what each does to weights
- [ ] Can explain K-Fold CV and why Stratified K-Fold matters for imbalance
- [ ] Can derive Sigmoid's purpose and explain log-odds intuitively
- [ ] Can build a confusion matrix and compute Precision/Recall/F1 from it
- [ ] Know why Accuracy fails on imbalanced data + know all 5 imbalance fixes
- [ ] Can explain KNN's lazy learning + why scaling is mandatory
- [ ] Can explain Gini/Entropy/Information Gain and how trees pick splits
- [ ] Can explain Bagging vs Boosting — parallel vs sequential, variance vs bias
- [ ] Can explain OOB error and why more trees ≠ more overfitting in Random Forest
- [ ] Can explain learning_rate/n_estimators tradeoff in Boosting
- [ ] Can walk through KMeans' Assignment + Update steps and Elbow Method
- [ ] Can explain PCA's purpose, explained variance, and PCA vs Feature Selection
- [ ] Can deliver a structured "walk me through an ML project" answer end-to-end
- [ ] Know the universal `fit()` / `predict()` pattern applies across every algorithm
- [ ] Can explain Model Drift and why Monitoring matters post-deployment