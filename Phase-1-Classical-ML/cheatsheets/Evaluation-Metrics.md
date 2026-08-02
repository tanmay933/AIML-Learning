# Evaluation Metrics

---

## Confusion Matrix (Foundation for All Classification Metrics)

```
                        Predicted Positive    Predicted Negative
Actual Positive         True Positive (TP)    False Negative (FN)
Actual Negative         False Positive (FP)   True Negative (TN)
```

| Term | Meaning |
|---|---|
| TP | Correctly predicted positive |
| TN | Correctly predicted negative |
| FP | Incorrectly predicted positive ("false alarm") |
| FN | Incorrectly predicted negative ("missed case") |

**Every classification metric below is a different combination of these four numbers.**

---

# REGRESSION METRICS

## MAE (Mean Absolute Error)

| Aspect | Detail |
|---|---|
| **Formula** | `MAE = average of \|actual − predicted\|` |
| **Intuition** | "What's the typical error size, in the target's own units?" |
| **Range** | 0 to ∞ (lower is better) |
| **Interpretation** | MAE = 12,000 → predictions off by $12,000 on average |
| **Advantages** | Robust to outliers, easy to interpret, same units as target |
| **Disadvantages** | Treats all errors equally — doesn't penalize large mistakes more |
| **When to Use** | Reporting to non-technical stakeholders; when all errors matter roughly equally |
| **Interview Q** | "Why prefer MAE over MSE?" → *"When you want a robust, easily interpretable error size and don't want large outliers to dominate the metric."* |

---

## MSE (Mean Squared Error)

| Aspect | Detail |
|---|---|
| **Formula** | `MSE = average of (actual − predicted)²` |
| **Intuition** | Same as MAE, but squares residuals — penalizes large errors much more |
| **Range** | 0 to ∞ (lower is better) |
| **Interpretation** | Units are squared (e.g., dollars²) — not directly interpretable |
| **Advantages** | Heavily penalizes large errors; mathematically convenient (differentiable) |
| **Disadvantages** | Very sensitive to outliers; hard to interpret due to squared units |
| **When to Use** | As a training cost function; when large errors are especially costly |
| **Interview Q** | "Why is MSE more outlier-sensitive than MAE?" → *"Squaring amplifies large residuals disproportionately — a residual of 20 contributes 100x more than a residual of 2."* |

---

## RMSE (Root Mean Squared Error)

| Aspect | Detail |
|---|---|
| **Formula** | `RMSE = √MSE` |
| **Intuition** | Brings MSE back into the target's original units while keeping its outlier sensitivity |
| **Range** | 0 to ∞ (lower is better) |
| **Interpretation** | Same units as target; RMSE ≥ MAE always |
| **Advantages** | Interpretable AND penalizes large errors — best of both MAE and MSE |
| **Disadvantages** | Still sensitive to outliers |
| **When to Use** | Most common go-to regression metric; when large errors matter more but interpretability is needed |
| **Interview Q** | "If RMSE >> MAE, what does that mean?" → *"A few large errors (outliers) are disproportionately inflating RMSE."* |

---

## R² (Coefficient of Determination)

| Aspect | Detail |
|---|---|
| **Formula** | `R² = 1 − (model's total squared error) / (baseline's total squared error)` — baseline = always predicting the mean |
| **Intuition** | "How much better is my model than just guessing the average every time?" |
| **Range** | −∞ to 1 (can be negative) |
| **Interpretation** | R²=0.85 → model explains 85% of the variance in the target |
| **Advantages** | Intuitive relative measure; easy to communicate as a percentage |
| **Disadvantages** | Can be misleading — high R² doesn't guarantee low absolute error; always increases when adding any feature, even useless ones |
| **When to Use** | Communicating overall explanatory power to stakeholders |
| **Interview Q** | "Can R² be negative? What does it mean?" → *"Yes — the model performs worse than predicting the mean every time, usually signaling a serious problem."* |

---

## Adjusted R²

| Aspect | Detail |
|---|---|
| **Formula** | R², penalized for the number of features relative to sample size (conceptual, not derived) |
| **Intuition** | Fixes R²'s flaw of always increasing with more features, even useless ones |
| **Range** | −∞ to 1 (can be lower than R²) |
| **Interpretation** | If R² rises but Adjusted R² falls when adding a feature → that feature isn't genuinely useful |
| **Advantages** | Fair comparison across models with different numbers of features |
| **Disadvantages** | Slightly less intuitive to explain than plain R² |
| **When to Use** | Comparing models with different feature counts |
| **Interview Q** | "Why does Adjusted R² exist?" → *"R² can only increase as features are added, regardless of usefulness — Adjusted R² penalizes unnecessary features."* |

### Regression Metrics Comparison Table

| Metric | Units | Outlier Sensitivity | Interpretability |
|---|---|---|---|
| MAE | Target units | Low | High |
| MSE | Squared units | Very High | Low |
| RMSE | Target units | High | High |
| R² | Unitless (%) | Moderate | High |
| Adjusted R² | Unitless (%) | Moderate | High |

---

# CLASSIFICATION METRICS

## Accuracy

| Aspect | Detail |
|---|---|
| **Formula** | `(TP + TN) / (TP + TN + FP + FN)` |
| **Intuition** | "Out of all predictions, what fraction were correct?" |
| **Range** | 0 to 1 |
| **Interpretation** | 0.945 → 94.5% of predictions were correct |
| **Advantages** | Simple, intuitive, works well on balanced data |
| **Disadvantages** | Highly misleading on imbalanced data — majority class dominates |
| **When to Use** | Roughly balanced classes, errors equally costly |
| **Interview Q** | "Why can 99% accuracy be a bad model?" → *"On imbalanced data (e.g., 1% fraud), always predicting the majority class hits 99% accuracy while catching zero fraud cases."* |

**Business Example**: A fraud model with 99% accuracy that never flags fraud is useless — check Recall instead.

---

## Precision

| Aspect | Detail |
|---|---|
| **Formula** | `TP / (TP + FP)` |
| **Intuition** | "When the model says positive, how often is it right?" |
| **Range** | 0 to 1 |
| **Interpretation** | 0.68 → 68% of flagged positives are actually correct |
| **Advantages** | Directly measures trustworthiness of positive predictions |
| **Disadvantages** | Ignores false negatives entirely — doesn't tell you what was missed |
| **When to Use** | False Positives are costly (e.g., spam filtering, marketing budget targeting) |
| **Interview Q** | "When would you prioritize Precision?" → *"When false positives are expensive — e.g., flagging a legitimate email as spam."* |

**Business Example**: Spam detection — a false positive means an important email gets buried.

---

## Recall (Sensitivity)

| Aspect | Detail |
|---|---|
| **Formula** | `TP / (TP + FN)` |
| **Intuition** | "Out of all actual positives, how many did we catch?" |
| **Range** | 0 to 1 |
| **Interpretation** | 0.85 → model catches 85% of actual positive cases |
| **Advantages** | Directly measures coverage of the positive/important class |
| **Disadvantages** | Ignores false positives — doesn't tell you about false alarms |
| **When to Use** | False Negatives are costly (e.g., disease screening, fraud detection) |
| **Interview Q** | "When would you prioritize Recall?" → *"When missing a real positive is far worse than a false alarm — e.g., cancer screening, fraud detection."* |

**Business Example**: Disease screening — missing an actual case (FN) is far worse than a false alarm.

---

## F1 Score

| Aspect | Detail |
|---|---|
| **Formula** | `2 × (Precision × Recall) / (Precision + Recall)` — harmonic mean |
| **Intuition** | A single number that stays high only when BOTH Precision and Recall are reasonably good |
| **Range** | 0 to 1 |
| **Interpretation** | 0.76 → balanced Precision/Recall performance |
| **Advantages** | Punishes extreme imbalance between Precision and Recall (unlike a simple average) |
| **Disadvantages** | Hides which of Precision/Recall is actually low; not always what business needs |
| **When to Use** | Need one balanced metric; imbalanced class distribution |
| **Interview Q** | "Why harmonic mean instead of average?" → *"Harmonic mean heavily penalizes cases where one metric is very low, unlike a simple average which can mask serious imbalance."* |

---

## Specificity

| Aspect | Detail |
|---|---|
| **Formula** | `TN / (TN + FP)` |
| **Intuition** | "Out of all actual negatives, how many did we correctly identify?" |
| **Range** | 0 to 1 |
| **Interpretation** | The negative-class counterpart to Recall |
| **Advantages** | Useful when correctly identifying negatives matters (e.g., confirming healthy patients) |
| **Disadvantages** | Rarely the primary metric — usually secondary to Recall |
| **When to Use** | Alongside Recall, when both correct classifications matter |
| **Interview Q** | "Relationship between Specificity and FPR?" → *"FPR = 1 − Specificity."* |

---

## ROC Curve

| Aspect | Detail |
|---|---|
| **Formula** | Plots True Positive Rate (Recall) vs False Positive Rate across ALL thresholds |
| **Intuition** | Shows the Recall/FPR tradeoff as the decision threshold changes |
| **Range** | Both axes: 0 to 1 |
| **Interpretation** | Curve bowing toward top-left = better; diagonal = random guessing |
| **Advantages** | Threshold-independent view of model performance |
| **Disadvantages** | Can look overly optimistic on imbalanced data (large TN count inflates it) |
| **When to Use** | Comparing models' overall ranking ability, balanced classes |
| **Interview Q** | "What does a diagonal ROC curve mean?" → *"The model performs no better than random guessing."* |

---

## ROC-AUC

| Aspect | Detail |
|---|---|
| **Formula** | Area under the ROC Curve |
| **Intuition** | Probability a random positive is ranked higher than a random negative |
| **Range** | 0 to 1 (0.5 = random, 1.0 = perfect) |
| **Interpretation** | AUC=0.85 → strong separability between classes |
| **Advantages** | Single number summarizing ranking quality across all thresholds |
| **Disadvantages** | Can be misleadingly high on heavily imbalanced datasets |
| **When to Use** | Comparing models independent of a specific threshold, balanced/moderately imbalanced data |
| **Interview Q** | "What does AUC actually measure?" → *"The probability the model ranks a random positive example higher than a random negative example."* |

---

## Precision-Recall Curve

| Aspect | Detail |
|---|---|
| **Formula** | Plots Precision vs Recall across all thresholds |
| **Intuition** | Focuses entirely on positive-class performance, ignoring TN count |
| **Range** | Both axes: 0 to 1 |
| **Interpretation** | Curve staying high across Recall values = strong performance on the minority class |
| **Advantages** | More honest than ROC on heavily imbalanced datasets |
| **Disadvantages** | Less familiar/standard than ROC; not ideal for balanced datasets |
| **When to Use** | Heavily imbalanced datasets, positive class is the focus (fraud, disease) |
| **Interview Q** | "PR Curve vs ROC — when to prefer PR?" → *"On heavily imbalanced data, since ROC's FPR is diluted by a large number of true negatives, making it look overly optimistic."* |

### Classification Metrics Comparison Table

| Metric | Formula | Focuses On | Best For |
|---|---|---|---|
| Accuracy | (TP+TN)/Total | Overall correctness | Balanced classes |
| Precision | TP/(TP+FP) | Positive prediction quality | FP costly (spam, marketing) |
| Recall | TP/(TP+FN) | Positive coverage | FN costly (disease, fraud) |
| F1 | Harmonic mean(P,R) | Balance | Imbalanced data, need one number |
| Specificity | TN/(TN+FP) | Negative coverage | Complements Recall |
| ROC-AUC | Area under ROC | Ranking, all thresholds | Balanced/moderate imbalance |
| PR-AUC | Area under PR curve | Positive-class ranking | Heavy imbalance |

---

## Metric Selection Flowchart

```
START
  │
  ▼
Is this a REGRESSION or CLASSIFICATION problem?
  │
  ├── REGRESSION
  │      │
  │      ▼
  │   Do outliers exist and should they be downweighted?
  │      │
  │      ├── YES ──► Use MAE
  │      └── NO ───► Use RMSE (or MSE during training)
  │      │
  │      ▼
  │   Need a relative "% explained" summary for stakeholders?
  │      │
  │      └── YES ──► Report R² (Adjusted R² if comparing feature counts)
  │
  └── CLASSIFICATION
         │
         ▼
     Are classes roughly balanced?
         │
         ├── YES ──► Accuracy is reasonable (report alongside F1/AUC)
         │
         └── NO ───► Do NOT use Accuracy alone
                       │
                       ▼
                 Which error is more costly?
                       │
                       ├── False Positives costlier ──► Prioritize Precision
                       ├── False Negatives costlier ──► Prioritize Recall
                       └── Both matter equally ────────► Use F1 Score
                       │
                       ▼
                 Comparing models across thresholds?
                       │
                       ├── Balanced/moderate imbalance ──► ROC-AUC
                       └── Heavy imbalance ────────────► Precision-Recall Curve
```

---

# Metric Selection Cheat Sheet

| Situation | Use This Metric |
|---|---|
| Reporting typical error size to a non-technical audience | MAE |
| Large errors are especially costly (regression) | RMSE / MSE |
| Comparing overall model "explanatory power" | R² |
| Comparing models with different feature counts | Adjusted R² |
| Balanced classification, errors equally costly | Accuracy |
| False Positives are expensive (spam, wasted budget) | Precision |
| False Negatives are expensive (disease, fraud, security) | Recall |
| Need one balanced classification number | F1 Score |
| Comparing classifiers' ranking ability (balanced data) | ROC-AUC |
| Comparing classifiers on heavily imbalanced data | Precision-Recall Curve / PR-AUC |
| Need to check both class performances explicitly | Confusion Matrix (always) |

**⭐ Golden Rules to Remember:**
- Never trust a single metric alone — always check the confusion matrix directly.
- Accuracy fails silently on imbalanced data.
- RMSE ≥ MAE always — a large gap signals outlier influence.
- R² can be negative — that's a valid (bad) result, not a bug.
- Precision and Recall trade off — improving one typically costs the other.
- ROC-AUC can look optimistic on imbalanced data — prefer PR-AUC there.