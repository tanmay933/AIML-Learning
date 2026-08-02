# Regression Metrics

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain why evaluation metrics are necessary beyond just training a model
- Interpret residuals and understand what a residual plot reveals
- Calculate and interpret MAE, MSE, RMSE, R², and Adjusted R²
- Explain the tradeoffs between each metric, including outlier sensitivity
- Choose the right metric for a given business problem
- Avoid common mistakes in reporting and interpreting regression performance
- Answer interview questions on regression evaluation with confidence

---

# Why This Topic Exists

Chapter 3 taught you how a Linear Regression model learns coefficients by minimizing a cost function. But **minimizing a cost function during training tells you nothing on its own about whether the model is actually good.**

You need a separate, standardized way to answer questions like:

- "How wrong is this model, on average?"
- "Is this model better than a naive guess?"
- "Should I report this performance to a non-technical stakeholder?"

That's what regression metrics are for. They're the vocabulary you'll use to judge, compare, and communicate about every regression model you build — not just Linear Regression, but every regression algorithm later in this handbook.

---

# Intuition

## Why Metrics Exist

💡 **Intuition**: Training a model answers *"what coefficients minimize error on this specific training data?"* Evaluation answers a different, more important question: *"how well does this model actually perform, in terms a human can understand and trust?"*

⭐ **Must Know**: **Training loss (the cost function used during fitting) and evaluation metrics are related but serve different purposes.** The cost function (e.g., MSE) drives *how the model learns*. Metrics are used *after* training to assess and communicate performance — often on the **test set**, using the same or different measures.

🚀 **Practical Insight**: A cost function like MSE is optimized mathematically because it's easy to differentiate — that's a computational convenience, not necessarily the most business-relevant way to describe performance. This is why metrics like MAE or R² are often reported to stakeholders even if MSE drove training internally.

📌 **Revision Point**: Always evaluate metrics on the **test set** (unseen data) — evaluating on training data only tells you how well the model memorized, not how well it generalizes (a direct callback to Chapter 1's concept of generalization).

---

# Core Concepts

## 1. Residuals

A residual is the gap between what actually happened and what the model predicted:

```
residual = actual − predicted
```

| Residual Value | Meaning |
|---|---|
| Positive | Model under-predicted (actual was higher) |
| Negative | Model over-predicted (actual was lower) |
| Near zero | Model was accurate for that point |

💡 **Intuition**: Every metric in this chapter is really just a different way of **summarizing the residuals** across all data points into one interpretable number.

### Residual Plots — Intuition

A residual plot puts predicted values on the x-axis and residuals on the y-axis:

```
Residual
   |
   |    *        *
   |  *    *  *      *
   |________________________ Predicted
   |     *    *   *
   |  *          *
```

⭐ **Must Know**: A **healthy** residual plot shows points scattered randomly around zero, with no visible pattern. If you see a clear shape — a curve, a funnel, a trend — it's a signal something is wrong (e.g., a non-linear relationship the model isn't capturing, or non-constant error variance). This connects directly to the **Linear Regression Assumptions** chapter, which covers this in depth — for now, just recognize the pattern.

---

## 2. Mean Absolute Error (MAE)

### Formula and Intuition

```
MAE = average of |actual − predicted|  across all samples
```

💡 **Intuition**: Take every residual, strip away its sign (so under- and over-predictions count equally), and average them. The result is: *"On average, how far off is the model, in the same units as the target?"*

**Example**: If MAE = 12,000 for a house price model, it means the model's predictions are, on average, off by $12,000 — in either direction.

### Advantages and Disadvantages

| Advantages | Disadvantages |
|---|---|
| Easy to interpret — same units as the target | Treats all errors equally — doesn't penalize large errors more |
| Robust to outliers (doesn't blow up due to a few extreme points) | Less mathematically convenient for optimization (not smooth at zero) |
| Good for reporting to non-technical stakeholders | Can hide the presence of a few very large, costly errors |

🎯 **Interview Tip**: MAE answers *"what's the typical error size?"* — it's the most intuitive metric to explain to a business audience.

---

## 3. Mean Squared Error (MSE)

### Formula and Intuition

```
MSE = average of (actual − predicted)²  across all samples
```

💡 **Intuition**: Same idea as MAE, but each residual is **squared** before averaging.

### Why Squaring?

- Squaring removes the sign problem (same as absolute value) — but also **penalizes large errors much more heavily** than small ones.
- A residual of 2 contributes `4` to the total. A residual of 20 contributes `400` — 100x more, not just 10x.

⚠ **Common Mistake**: Forgetting that MSE's units are the **square** of the target's units (e.g., dollars²), which makes it much harder to interpret directly — this is precisely why RMSE exists (next section).

### Sensitivity to Outliers

⭐ **Must Know**: MSE is **highly sensitive to outliers** because of the squaring operation. A single very bad prediction can dominate the entire metric, making the model look far worse overall than it typically performs.

| Scenario | MAE Impact | MSE Impact |
|---|---|---|
| One prediction is off by 100 (rest are off by ~2) | Modest increase | Large increase — dominates the metric |

🎯 **Interview Tip**: If asked "MAE vs MSE, which is more sensitive to outliers?" — the answer is MSE, because squaring disproportionately amplifies large residuals.

---

## 4. Root Mean Squared Error (RMSE)

### Formula and Relationship to MSE

```
RMSE = √MSE
```

💡 **Intuition**: RMSE takes MSE and brings it back into the **original units** of the target by undoing the squaring with a square root. This makes it far easier to interpret than raw MSE, while still keeping MSE's "penalize large errors more" behavior.

**Example**: If MSE = 144,000,000 (dollars²), RMSE = 12,000 (dollars) — directly comparable to MAE.

### Interpretation

| Metric | Units | Penalizes Large Errors? | Easy to Interpret? |
|---|---|---|---|
| MSE | Squared units (e.g., $²) | Yes, heavily | No |
| RMSE | Original units (e.g., $) | Yes, heavily | Yes |

⭐ **Must Know**: RMSE and MAE are both in the same units as the target, but **RMSE will always be ≥ MAE** for the same dataset. The bigger the gap between RMSE and MAE, the more the errors are dominated by a few large residuals (outliers).

📌 **Revision Point**: RMSE ≈ MAE → errors are fairly uniform in size. RMSE >> MAE → a few large errors are dragging performance down.

---

## 5. R² Score (Coefficient of Determination)

### Intuition

💡 **Intuition**: R² answers a very different question from MAE/MSE/RMSE: *"How much better is my model compared to just predicting the average every time?"*

### The Baseline Model

Imagine the simplest possible "model": ignore all features, and always predict the **mean** of the target for every single sample. This is the baseline R² is measured against.

```
R² = 1 − (model's total squared error) / (baseline's total squared error)
```

We are not deriving this — just understand what it compares.

### Interpreting Values

| R² Value | Meaning |
|---|---|
| 1.0 | Perfect predictions — model explains 100% of the variance in the target |
| 0.0 | Model performs no better than always predicting the mean |
| Negative | Model performs **worse** than just predicting the mean (a bad sign) |
| 0.75 | Model explains 75% of the variance in the target — a common way this is reported |

**Example**: An R² of 0.85 for a house price model means the model explains 85% of the variation in house prices using its features — the remaining 15% is unexplained (noise, missing features, etc.).

⚠ **Common Mistake**: Assuming R² alone tells you if a model is "good." A high R² can still come with large absolute errors if the target has huge variance. R² measures *relative* explanatory power, not absolute prediction accuracy — that's what MAE/RMSE are for.

🎯 **Interview Tip**: If asked "Can R² be negative?" — yes. It means the model is doing worse than a naive average-based guess, which usually signals something is seriously wrong (bad features, wrong algorithm, or a bug in the pipeline).

---

## 6. Adjusted R²

### Why R² Is Insufficient

⭐ **Must Know**: R² has a subtle flaw: **it can only increase (or stay the same) as you add more features — even completely useless, random ones.** This makes plain R² unreliable for comparing models with different numbers of features.

💡 **Intuition**: Adding any feature gives the model one more "knob" to fit the training data slightly better, even if that feature has no real predictive value. R² doesn't penalize this — it just rewards the (possibly fake) improvement.

### The Penalty for Unnecessary Features

Adjusted R² modifies R² by **penalizing the addition of features that don't meaningfully improve the model**, taking into account both the number of features and the number of samples.

| Metric | Increases when you add... |
|---|---|
| R² | Any feature, useful or not |
| Adjusted R² | Only features that improve the model more than random chance would predict |

⭐ **Must Know**: If you add a useless feature and R² goes up but **Adjusted R² goes down**, that's a clear sign the feature isn't actually helping — it's noise.

🎯 **Interview Tip**: If asked "When would you prefer Adjusted R² over R²?" — whenever comparing models with **different numbers of features**, since plain R² unfairly favors models with more features regardless of whether they're useful. (Feature selection strategies themselves are covered in later chapters.)

---

## 7. Metric Comparison

| Metric | Units | Outlier Sensitivity | Interpretability | What It Answers |
|---|---|---|---|---|
| **MAE** | Same as target | Low | High | "What's the typical error size?" |
| **MSE** | Squared target units | Very High | Low | "How large are errors, penalizing big mistakes heavily?" |
| **RMSE** | Same as target | High | High | "What's the typical error size, with big mistakes weighted more?" |
| **R²** | Unitless (0–1 scale, can go negative) | Moderate | High (as a %) | "How much better is my model than a naive average guess?" |
| **Adjusted R²** | Unitless | Moderate | High (as a %) | "How much better is my model, accounting for feature count?" |

### When to Prefer Each

| Situation | Preferred Metric |
|---|---|
| Explaining error size to a non-technical stakeholder | MAE |
| Large errors are especially costly and must be penalized | RMSE or MSE |
| Comparing overall explanatory power of a model | R² |
| Comparing models with different numbers of features | Adjusted R² |
| Outliers are known data errors you want to downweight | MAE (more robust) |
| Outliers are important and must be strongly penalized | RMSE/MSE |

---

## 8. Choosing the Right Metric

Metric choice should be driven by the **business problem**, not just mathematical convenience.

| Business Scenario | Recommended Metric | Reasoning |
|---|---|---|
| Predicting delivery time for a logistics app | MAE | Every minute of error matters similarly; easy to communicate ("off by ~5 minutes on average") |
| Predicting insurance claim amounts | RMSE | Large mispredictions (e.g., missing a huge claim) are far more costly — should be penalized heavily |
| Reporting overall model quality to leadership | R² | Gives a quick, intuitive "how good is this model overall" summary |
| Comparing two models — one with 5 features, one with 20 | Adjusted R² | Prevents being misled by R² inflation from extra features |
| House price prediction, where a few mansions are extreme outliers | MAE (or careful outlier handling first) | Prevents a handful of extreme homes from dominating the evaluation |

🚀 **Practical Insight**: In real projects, it's common to report **more than one metric together** — e.g., "RMSE = $14,000, R² = 0.82" — since each tells a different part of the story. Relying on a single metric can be misleading.

---

## 9. Common Beginner Mistakes

| Mistake | Consequence |
|---|---|
| **Evaluating only on training data** | Metrics look artificially good; doesn't reflect real-world generalization |
| **Reporting only R²** | Hides the actual magnitude of errors — a high R² can still mean large absolute mistakes if the target has high variance |
| **Using MSE/RMSE without checking for outliers first** | A few extreme points can make the model look far worse than its typical performance |
| **Comparing R² across models with different feature counts** | Misleading — more features inflate R² regardless of usefulness; use Adjusted R² instead |
| **Assuming a negative R² is a bug** | It's a valid (if bad) outcome — it simply means the model underperforms a naive mean-based guess |
| **Ignoring residual plots** | Misses systematic patterns (e.g., non-linearity) that a single summary metric can't reveal |

---

## 10. Interview Tips

**Q: What's the difference between MAE and MSE?**
> MAE averages the absolute value of residuals, treating all errors proportionally. MSE averages the squared residuals, which penalizes larger errors much more heavily and is more sensitive to outliers.

**Q: Why do we use RMSE instead of just MSE?**
> RMSE takes the square root of MSE, bringing the error back into the same units as the target variable, which makes it far more interpretable while still retaining MSE's sensitivity to large errors.

**Q: What does an R² of 0.9 mean?**
> The model explains 90% of the variance in the target variable compared to a naive baseline that always predicts the mean. It does not directly tell you the size of the errors in absolute terms.

**Q: Can R² be negative? What does that mean?**
> Yes. A negative R² means the model performs worse than simply predicting the average value every time — usually a sign of a poorly fit model, wrong features, or a pipeline issue.

**Q: Why does Adjusted R² exist if we already have R²?**
> R² can only increase as more features are added, even if those features are irrelevant. Adjusted R² penalizes unnecessary features, making it more reliable when comparing models with different numbers of features.

**Q: Which metric is most robust to outliers: MAE, MSE, or RMSE?**
> MAE, because it uses absolute value rather than squaring, so it doesn't disproportionately amplify large errors the way MSE and RMSE do.

**Q: If RMSE is much higher than MAE, what does that suggest?**
> It suggests the presence of a few large errors (outliers) that are disproportionately inflating RMSE, since RMSE penalizes large residuals more heavily than MAE does.

**Q: How would you choose between MAE and RMSE for a real project?**
> It depends on whether large errors are especially costly for the business. If all errors matter roughly equally, use MAE. If large errors are far more damaging (e.g., in finance or safety-critical predictions), use RMSE to appropriately penalize them.

---

# Quick Revision

## Formula Summary

| Metric | Formula (conceptual) |
|---|---|
| MAE | average of \|actual − predicted\| |
| MSE | average of (actual − predicted)² |
| RMSE | √MSE |
| R² | 1 − (model error / baseline mean-prediction error) |
| Adjusted R² | R², penalized for number of features relative to sample size |

## Interpretation Table

| Metric | Good Value Looks Like | Bad Value Looks Like |
|---|---|---|
| MAE | Small, close to 0, in target's units | Large relative to target's typical scale |
| RMSE | Small, close to MAE | Much larger than MAE (signals outlier influence) |
| R² | Close to 1.0 | Close to 0 or negative |
| Adjusted R² | Close to R², not dropping as features increase | Noticeably lower than R² (unnecessary features present) |

## Interview Facts Cheat Sheet

- MAE = average absolute error → robust, easy to interpret.
- MSE = average squared error → penalizes large errors, sensitive to outliers, hard to interpret directly (squared units).
- RMSE = √MSE → same interpretability as MAE, same outlier sensitivity as MSE.
- R² compares your model to a "always predict the mean" baseline; can be negative.
- Adjusted R² penalizes useless added features; use it when comparing models with different feature counts.
- Always evaluate on the test set, never just training data.
- Report multiple metrics together — no single metric tells the whole story.

## ✅ Checklist: "I understand this chapter if I can..."

- [ ] Explain why training loss and evaluation metrics serve different purposes
- [ ] Define a residual and describe what a healthy residual plot looks like
- [ ] Calculate MAE and MSE conceptually from a small set of residuals
- [ ] Explain why MSE is more sensitive to outliers than MAE
- [ ] Explain why RMSE exists instead of just using MSE
- [ ] Interpret an R² value, including what a negative R² means
- [ ] Explain why R² alone can be misleading when comparing models with different feature counts
- [ ] Choose the most appropriate metric for a given business scenario
- [ ] List at least 4 common mistakes in regression evaluation
- [ ] Answer every interview question in this chapter without looking