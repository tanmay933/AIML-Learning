# Linear Regression Assumptions

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain why Linear Regression relies on a set of underlying statistical assumptions
- Define linearity, independence, homoscedasticity, normality of residuals, and no multicollinearity
- Detect violations of each assumption using practical, visual methods
- Explain what happens to a model's reliability when each assumption is violated
- Distinguish correlation from causation and avoid a classic interpretation trap
- Recognize why a model can have decent metrics yet still be statistically unreliable
- Answer interview questions about Linear Regression assumptions with engineering-grade clarity

---

# Why This Topic Exists

Chapter 3 showed you *how* Linear Regression fits a line. Chapter 4 showed you *how to measure* whether that line predicts well. This chapter answers a different, more subtle question: **can you actually trust the model's coefficients and predictions?**

Linear Regression isn't just a curve-fitting trick — it's built on a set of statistical assumptions about how the data behaves. When those assumptions hold, the model's coefficients are reliable and its errors behave predictably. When they don't, you can get a model that scores reasonably on MAE/RMSE/R² but is quietly misleading — unstable coefficients, untrustworthy confidence in predictions, or systematic blind spots in certain regions of the data.

This is exactly the "why does Linear Regression sometimes perform poorly?" question — and it's frequently the gap between a beginner who can call `.fit()` and an engineer who understands what the model is actually doing.

---

# Intuition

💡 **Intuition**: Think of Linear Regression's assumptions as the "terms and conditions" of the algorithm. It never checks them for you — it will happily fit a line to *any* data, even data where a straight line makes no sense, or where the errors behave in wildly inconsistent ways. The assumptions describe the conditions under which the model's output can be **trusted**, not just computed.

⭐ **Must Know**: **Violating an assumption doesn't always mean the model crashes or produces obviously bad predictions.** Often it silently degrades reliability — coefficients become unstable, error estimates become misleading, or performance quietly varies across different regions of the input space. This is what makes assumption-checking an engineering discipline, not just a formality.

---

# Core Concepts

## 1. Linearity

**What it means**: The relationship between each feature and the target is assumed to be approximately a straight-line (linear) relationship — not curved, not exponential, not cyclical.

**Why it matters**: The entire model is a weighted sum of features. If the true relationship curves, a straight line can only approximate part of it well — it will systematically miss the pattern elsewhere.

**What happens if violated**: The model **underfits** — it consistently over-predicts in some regions and under-predicts in others, because a straight line simply cannot bend to match a curved trend.

**How it's detected**:
- Plot each feature against the target — look for obvious curvature.
- Plot **residuals vs predicted values** (introduced in Chapter 4) — a clear curved pattern (not random scatter) signals non-linearity.

```
Residual
   |     *
   |   *   *
   | *       *
   |___________________ Predicted
   |*                 *
   |  *             *
   |     *   *   *
       (curved pattern → non-linearity)
```

**Possible remedies**:
- Transform the feature (e.g., log, square root) to make the relationship more linear.
- Use **Polynomial Regression** to capture curvature — covered in the next chapter; not explained in depth here.
- Switch to a non-linear algorithm (Decision Trees, later chapters) if linearity truly doesn't hold.

🎯 **Interview Tip**: If asked "How do you check if Linear Regression is appropriate?" — start with: *"Plot residuals vs predicted values; a random scatter around zero suggests linearity holds, while a visible pattern suggests it doesn't."*

---

## 2. Independence of Observations

**What it means**: Each data point's error is assumed to be unrelated to any other data point's error — knowing one residual should give you no information about another.

**Why it matters**: If errors are correlated (e.g., consecutive rows influence each other), the model's confidence in its own coefficients becomes overstated — it "thinks" it has more independent evidence than it actually does.

**What happens if violated**: Most common in **time-series data**, where today's error is often correlated with yesterday's error (a phenomenon called autocorrelation). The model may look statistically strong while actually being unreliable for forecasting.

**How it's detected**:
- Plot residuals **in the order they were collected** (e.g., by time or by sequence) — a clear pattern (trending up/down, cycling) signals dependence.
- This is especially relevant if your dataset has a natural order (time, geography, grouped by user/session).

```
Residual
   |        *  *
   |     *        *
   |  *              *
   |________________________ Time / Order
   |               *      *
   |            *
      (wave-like pattern → non-independence)
```

**Possible remedies**:
- Use models designed for sequential/time-dependent data (out of scope for this handbook, but good to be aware of).
- Ensure data isn't grouped in a way that leaks structure (e.g., multiple rows from the same user without accounting for that grouping).

📌 **Revision Point**: Independence issues are a major reason plain Linear Regression is a poor default choice for raw time-series forecasting.

---

## 3. Homoscedasticity

**What it means**: The size of the residuals (errors) should stay roughly **constant** across all levels of the predicted value — the "spread" of errors shouldn't grow or shrink as predictions get larger or smaller.

💡 **Intuition**: Imagine predicting house prices. If the model is equally accurate ($5,000 typical error) whether the house costs $100,000 or $1,000,000, that's homoscedastic. If errors get much bigger for expensive houses (e.g., $5,000 error for cheap homes, but $80,000 error for expensive ones), that's the opposite — **heteroscedasticity**.

**Why it matters**: Homoscedasticity underlies the reliability of confidence intervals and statistical inference on coefficients. When violated, the model tends to fit some regions of the data much better than others, without that inconsistency being visible in a single overall metric like RMSE.

**What happens if violated**: Predictions become **unreliable in certain ranges** — often the model is decent for typical values but notably worse for extreme ones. Aggregate metrics (Chapter 4) can hide this problem entirely.

**How it's detected**:
- Plot residuals vs predicted values — look for a **funnel shape** (errors fanning out or narrowing) rather than a uniform band.

```
Residual
   |  *
   |  *  *
   |  *  *   *
   |__*__*___*____*_______ Predicted
   |  *  *   *      *
   |  *  *          *  *
   (funnel shape → heteroscedasticity)
```

**Possible remedies**:
- Transform the target variable (e.g., log transform), which often stabilizes variance.
- Consider whether the model needs to weight different regions of data differently, or whether a different algorithm is more appropriate.

🎯 **Interview Tip**: A funnel-shaped residual plot is the classic visual signature of heteroscedasticity — this is one of the most commonly tested visual-interpretation questions in ML interviews.

---

## 4. Normality of Residuals

**What it means**: The residuals (errors), taken as a whole, are assumed to roughly follow a **normal (bell-curve) distribution** centered at zero.

**Why it matters**: This assumption primarily supports the statistical reliability of confidence intervals and hypothesis tests on the coefficients (e.g., "is this feature's effect statistically significant?"). It matters less for raw prediction accuracy and more for **trusting the model's statistical conclusions**.

**What happens if violated**: Confidence intervals and significance tests on coefficients become unreliable — you might think a feature's effect is statistically meaningful (or not) when it actually isn't. Point predictions themselves are often still reasonably usable even with mild violations.

**How it's detected**:
- Plot a **histogram of residuals** — should look roughly bell-shaped and centered around zero.
- A **Q-Q plot** (quantile-quantile plot) is the more formal tool — points should roughly follow a straight diagonal line if residuals are normal. (Just know this exists; no need to derive it.)

```
Frequency
   |        ___
   |      _/   \_
   |    _/       \_
   |  _/           \_
   |_/_______________\____ Residual value
              0
      (roughly bell-shaped → normal)
```

**Possible remedies**:
- Transform the target variable.
- Investigate whether outliers are distorting the residual distribution.
- Accept the deviation if it's mild — this is often the least critical assumption for pure prediction tasks, more critical for formal statistical inference.

📌 **Revision Point**: Of all five assumptions, normality of residuals matters most when you care about **statistical inference** (p-values, confidence intervals on coefficients) — less so if you only care about raw predictive performance.

---

## 5. No Multicollinearity

**What it means**: Features used in the model should not be **highly correlated with each other**. Each feature should contribute distinct, independent information.

💡 **Intuition**: If `Area (sqft)` and `Area (sq meters)` were both included as features, they carry the *exact same information* in different units. The model can't tell how to split credit between them — small changes in data can wildly swing which one "gets credit," even though the total prediction barely changes.

**Why it matters**: When features are highly correlated, individual coefficients become **unstable and hard to interpret** — a small change in the data can flip a coefficient's sign or magnitude, even though overall prediction quality barely changes.

**What happens if violated**: Coefficient interpretation (Chapter 3's "holding other features constant" logic) breaks down, because it's no longer clear which correlated feature is "really" driving the effect. Predictions can still be reasonably accurate even when multicollinearity is present — it's primarily an **interpretability** problem, not always an accuracy problem.

**How it's detected**:
- A **correlation matrix** between features — very high correlation (e.g., >0.8–0.9) between two features is a red flag.
- **Variance Inflation Factor (VIF)** — a more formal diagnostic (just know the name and purpose; no need to compute it by hand here).

```
             Area   Bedrooms   AreaSqM
Area          1.0     0.65       0.99   ← Area and AreaSqM are nearly identical info
Bedrooms      0.65    1.0        0.64
AreaSqM       0.99    0.64       1.0
```

**Possible remedies**:
- Drop one of the correlated features.
- Combine correlated features into a single engineered feature (Chapter 2 concept).
- Use **Regularization** (Ridge/Lasso) to stabilize coefficients in the presence of correlated features — covered in a later chapter, only mentioned here as a preview.

⚠ **Common Mistake**: Assuming multicollinearity always hurts prediction accuracy. It mainly damages **coefficient interpretability and stability** — a model can still predict reasonably well even with multicollinear features present.

---

## 6. Residual Plots — The Universal Diagnostic Tool

⭐ **Must Know**: Notice that **four of the five assumptions above are all checked using some form of residual plot.** This is the single most important practical skill from this chapter.

| Plot Type | What It Diagnoses |
|---|---|
| Residuals vs Predicted Values | Linearity (pattern) and Homoscedasticity (funnel shape) |
| Residuals vs Order/Time | Independence of observations |
| Histogram / Q-Q plot of Residuals | Normality of residuals |
| Correlation matrix / VIF | Multicollinearity (not a residual plot, but the equivalent diagnostic) |

🚀 **Practical Insight**: In real projects, the residual-vs-predicted plot is usually the **first thing** an engineer checks after fitting a Linear Regression model — it catches both linearity and homoscedasticity problems in a single glance.

---

## 7. Correlation vs Causation

💡 **Intuition**: Linear Regression can tell you that two variables move together — it cannot tell you that one *causes* the other.

**Classic example**: Ice cream sales and drowning incidents are strongly correlated. Ice cream doesn't cause drowning — both increase because of a third factor: hot weather (more swimming, more ice cream purchases).

⭐ **Must Know**: A statistically significant, strong coefficient only tells you the feature is a good **predictor** — it says nothing about whether changing that feature would actually **cause** a change in the target in the real world. Confusing the two leads to bad business decisions (e.g., "increase ice cream sales to reduce drowning" — nonsensical, but structurally the same mistake people make with real features).

🎯 **Interview Tip**: This is one of the most common conceptual interview questions in all of ML. Always be ready to give an example (ice cream/drowning, or a domain-relevant one) when explaining correlation vs causation.

---

## 8. What Happens When Assumptions Fail — Summary

| Assumption Violated | Primary Consequence |
|---|---|
| **Linearity** | Systematic underfitting; model misses real patterns |
| **Independence** | Overconfident, unreliable coefficients; common in time-series data |
| **Homoscedasticity** | Inconsistent accuracy across different prediction ranges |
| **Normality of Residuals** | Unreliable confidence intervals/statistical significance on coefficients |
| **No Multicollinearity** | Unstable, hard-to-interpret coefficients |

⚠ **Common Mistake**: Assuming that if the model's overall metrics (Chapter 4: MAE, RMSE, R²) look fine, the assumptions must be satisfied. **Aggregate metrics can look perfectly reasonable even when assumptions are clearly violated** — this is exactly why dedicated diagnostic plots exist separately from the standard metrics.

---

## 9. Practical Examples

| Scenario | Likely Violation | Why |
|---|---|---|
| Predicting house price using `Area` and `Area in sq meters` | Multicollinearity | Both features encode near-identical information |
| Predicting daily stock returns using yesterday's data | Independence | Financial time-series data is often autocorrelated |
| Predicting income across a huge income range ($20K–$2M) | Homoscedasticity | Prediction error likely grows much larger for very high incomes |
| Predicting salary purely from years of experience, ignoring diminishing returns at senior levels | Linearity | The true relationship likely flattens out over time, not a straight line |

---

## 10. Common Misconceptions

| Misconception | Reality |
|---|---|
| "If R² is high, the assumptions must be fine" | Assumptions and predictive metrics are independent checks — a high R² can coexist with clear assumption violations |
| "Multicollinearity always ruins predictions" | It mainly damages coefficient interpretability and stability, not necessarily raw prediction accuracy |
| "Non-normal residuals mean the model is useless" | It primarily affects the reliability of statistical inference (confidence intervals, p-values), less so raw predictions |
| "Assumption checks are optional academic exercises" | In practice, they explain *why* a model underperforms or behaves unpredictably in specific situations — this is directly useful engineering knowledge |
| "A strong coefficient proves causation" | It only proves a statistical association — never assume causation from a coefficient alone |

---

## 11. Interview Tips

**Q: Why does Linear Regression rely on assumptions in the first place?**
> The model's coefficients and statistical guarantees (confidence intervals, significance tests) are only reliable when the data reasonably satisfies these conditions. The assumptions define when you can trust the model beyond just its raw predictions.

**Q: How would you check if the linearity assumption holds?**
> Plot residuals against predicted values. A random scatter around zero suggests linearity holds; a clear curved pattern suggests the true relationship is non-linear.

**Q: What is heteroscedasticity, and how do you detect it?**
> Heteroscedasticity is when the variance of residuals changes across the range of predicted values — often visible as a funnel shape in a residuals-vs-predicted plot. It means the model is more accurate in some ranges than others.

**Q: What is multicollinearity, and why is it a problem?**
> Multicollinearity is when two or more features are highly correlated with each other. It makes individual coefficients unstable and difficult to interpret, since the model can't clearly separate each feature's independent contribution — though it doesn't necessarily hurt overall prediction accuracy.

**Q: Does violating an assumption always make the model unusable?**
> No. Some violations (like mild non-normality of residuals) mainly affect statistical inference rather than prediction quality. Others (like non-linearity) can seriously degrade predictive accuracy. The impact depends on which assumption is violated and how severely.

**Q: What's the difference between correlation and causation?**
> Correlation means two variables move together statistically. Causation means one variable directly produces a change in the other. Linear Regression coefficients only demonstrate correlation/association — establishing causation requires additional evidence beyond the model itself (e.g., controlled experiments).

**Q: If your residual plot shows a funnel shape, what does that tell you?**
> It indicates heteroscedasticity — the model's error is inconsistent across different prediction ranges, meaning some regions of the data are predicted much less reliably than others.

---

# Quick Revision

## Assumptions Summary Table

| Assumption | Checked With | If Violated |
|---|---|---|
| Linearity | Residuals vs Predicted plot | Systematic underfitting |
| Independence | Residuals vs Time/Order plot | Overconfident, unreliable coefficients |
| Homoscedasticity | Residuals vs Predicted plot (funnel shape) | Inconsistent accuracy across prediction ranges |
| Normality of Residuals | Histogram / Q-Q plot of residuals | Unreliable confidence intervals & significance tests |
| No Multicollinearity | Correlation matrix / VIF | Unstable, hard-to-interpret coefficients |

## Terminology Recap

- **Residual** — actual value minus predicted value (from Chapter 4)
- **Heteroscedasticity** — non-constant variance of residuals
- **Autocorrelation** — residuals correlated with each other across order/time
- **Multicollinearity** — high correlation between input features
- **VIF (Variance Inflation Factor)** — formal diagnostic for multicollinearity
- **Q-Q Plot** — visual tool for checking if residuals are normally distributed

## Interview Facts Cheat Sheet

- Four of the five assumptions are diagnosed primarily through residual plots.
- A funnel-shaped residual plot = heteroscedasticity.
- A curved residual pattern = non-linearity.
- Wave-like residuals over time/order = non-independence.
- Multicollinearity hurts interpretability more than accuracy.
- Correlation ≠ causation — always.
- Good aggregate metrics (R², RMSE) do NOT guarantee assumptions are satisfied.

## ✅ Checklist: "I understand this chapter if I can..."

- [ ] Explain why Linear Regression assumptions matter beyond just prediction accuracy
- [ ] Define linearity, independence, homoscedasticity, normality, and multicollinearity in plain language
- [ ] Identify which residual plot detects which assumption violation
- [ ] Sketch (or describe) what a funnel-shaped residual plot looks like and what it means
- [ ] Explain what happens to coefficients when multicollinearity is present
- [ ] Explain why a high R² doesn't guarantee assumptions are satisfied
- [ ] Give a clear example distinguishing correlation from causation
- [ ] Explain why time-series data often violates the independence assumption
- [ ] Answer every interview question in this chapter without looking