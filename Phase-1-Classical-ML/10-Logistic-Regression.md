# Logistic Regression

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain why Linear Regression is unsuitable for classification problems
- Understand probability, odds, and log-odds, and how they connect to Logistic Regression
- Explain the Sigmoid function and why it's used
- Read and interpret the Logistic Regression equation
- Understand the decision boundary and how classification decisions are made
- Distinguish binary from multi-class classification at a high level
- Understand, at a high level, how a Logistic Regression model is trained
- Identify where Logistic Regression works well and where it struggles
- Answer interview questions on Logistic Regression with engineering-grade clarity

---

# Why This Topic Exists

Every algorithm so far in this handbook — Linear, Polynomial, Regularized Regression — has predicted a **continuous number**. But Chapter 1 told you that a huge category of real-world problems isn't about predicting a number at all — it's about predicting a **category**: spam or not spam, fraud or not fraud, will churn or won't churn.

Logistic Regression is the natural entry point into classification. Despite its name, it's not a regression algorithm in the sense you've learned — it's the **first classification algorithm** in this handbook, and it builds directly on the linear equation you already know from Chapter 3. Understanding it well sets up everything that follows: Classification Metrics, KNN, Decision Trees, and beyond.

---

# Intuition

## Why Linear Regression Cannot Solve Classification

💡 **Intuition**: Imagine trying to predict "Will this customer churn?" (Yes/No) using Linear Regression. You'd encode `Yes = 1`, `No = 0`, and fit a line — but a line has no boundaries. It can output `1.4`, `-0.3`, or `50` — numbers that have no meaningful interpretation as a yes/no decision.

```
Linear Regression forced onto a classification problem:

Churned (1)  |                              *  *
             |                         *
             |                    *
             |               *
             |          *
Not (0)      |_____*____________________________ Feature
             (line keeps going past 0 and 1 — meaningless output)
```

⚠ **Common Mistake**: Thinking you can just "round" a Linear Regression output to 0 or 1 and call it classification. This technically produces a decision, but the underlying line is a poor fit for this kind of data — it's sensitive to outliers (Chapter 3) and doesn't naturally produce anything resembling a **probability**, which is usually what you actually want from a classifier.

⭐ **Must Know**: What we actually need is a model that outputs a value **always between 0 and 1** — something interpretable as "the probability this belongs to the positive class." That's exactly what Logistic Regression is built to do.

---

# Core Concepts

## 1. Classification Problems

📌 **Revision Point**: Recall from Chapter 1 — classification predicts a **category**, not a continuous number. Logistic Regression, despite the name, is a classification algorithm — it just borrows the linear equation structure from regression and transforms it to produce valid probabilities.

## 2. Logistic Regression Intuition

💡 **Intuition**: Logistic Regression takes the familiar linear equation (a weighted sum of features), but instead of using that number directly as the prediction, it **squashes** it through a special function that forces the output into the range **[0, 1]** — a valid probability.

```
Linear Regression:    raw number  →  prediction directly
Logistic Regression:  raw number  →  squashing function  →  probability (0 to 1)
```

Once you have a probability, classification becomes simple: pick a threshold (commonly 0.5) — above it, predict "Class 1"; below it, predict "Class 0."

---

## 3. Probability

**What it is**: A number between 0 and 1 representing how likely an event is to happen. `0.8` means an 80% chance.

Logistic Regression's core output isn't actually "spam" or "not spam" — it's **"the probability this email is spam is 0.87."** The final class label is a decision made *from* that probability, not the model's raw output.

---

## 4. Odds

💡 **Intuition**: Odds reframe probability as a **ratio** — "how many times more likely is this to happen than not?"

```
Odds = P(event) / (1 - P(event))
```

**Example**: If P(spam) = 0.8, then odds = 0.8 / 0.2 = **4** — the email is 4 times more likely to be spam than not.

| Probability | Odds |
|---|---|
| 0.5 | 1 (equally likely) |
| 0.8 | 4 (much more likely to happen) |
| 0.2 | 0.25 (much more likely NOT to happen) |

📌 **Revision Point**: Probability lives in [0, 1]. Odds live in [0, ∞) — this range shift matters for the next step.

---

## 5. Log-Odds (Logit)

💡 **Intuition**: Taking the **logarithm** of odds stretches that [0, ∞) range into the full range of **all real numbers** (−∞ to +∞) — which conveniently matches the range of the plain linear equation (`w1x1 + w2x2 + ... + b`) from Chapter 3.

```
Log-Odds (Logit) = log(Odds) = log( P / (1-P) )
```

⭐ **Must Know**: This is the key trick that connects Logistic Regression back to the linear equation you already know: **Logistic Regression models the log-odds as a linear combination of features** — not the probability directly.

```
log( P / (1-P) ) = w1*x1 + w2*x2 + ... + wn*xn + b
```

We're not deriving this — just understand *why* it's structured this way: the right-hand side (linear equation) can be any real number, and log-odds is the one transformation of probability that also spans all real numbers. They're a natural match.

---

## 6. Sigmoid Function

**What it does**: Reverses the log-odds transformation — takes any real number (the output of the linear equation) and squashes it back into a valid probability between 0 and 1.

```
Sigmoid(z) = 1 / (1 + e^(-z))
```

💡 **Intuition**: You don't need to memorize this formula — just understand its **shape**:

```
Probability
   1 |                    _______________
     |                 __/
     |               _/
 0.5 |             _/
     |          __/
     |     ____/
   0 |____/
     |________________________________ z (linear equation output)
          -∞          0            +∞
```

- Very large positive `z` → Sigmoid output approaches **1** (very confident "Class 1")
- Very large negative `z` → Sigmoid output approaches **0** (very confident "Class 0")
- `z = 0` → Sigmoid output is exactly **0.5** (maximum uncertainty)

⭐ **Must Know**: The Sigmoid function is what makes Logistic Regression's output always a valid, interpretable probability — regardless of how large or small the underlying linear equation's output is.

🎯 **Interview Tip**: If asked "Why does Logistic Regression use the Sigmoid function?" — answer: *"To transform the unbounded output of a linear equation into a probability between 0 and 1, since raw linear outputs aren't valid probabilities."*

---

## 7. Logistic Regression Equation

Putting it all together:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b        (same linear equation as Chapter 3)

P(Class 1) = Sigmoid(z) = 1 / (1 + e^(-z))
```

| Symbol | Meaning |
|---|---|
| `x1, x2, ..., xn` | Input features |
| `w1, w2, ..., wn` | Learned coefficients (same role as Chapter 3, but now they influence log-odds, not the prediction directly) |
| `b` | Intercept |
| `z` | The linear combination — the "log-odds" |
| `P(Class 1)` | The final predicted probability, between 0 and 1 |

⭐ **Must Know**: **Coefficient interpretation changes from Linear Regression.** A coefficient `w` no longer means "the target increases by `w` units." Instead, it means "a one-unit increase in this feature increases the **log-odds** of the positive class by `w`" — a less directly intuitive, but still meaningful, relationship. In practice, engineers often interpret the **sign** of the coefficient (positive = increases likelihood of Class 1, negative = decreases it) more than the exact magnitude.

---

## 8. Decision Boundary

**What it is**: The threshold at which the model switches its predicted class — most commonly, `P ≥ 0.5 → Class 1`, `P < 0.5 → Class 0`.

```
Probability
   1 |                    _______________
     |                 __/
     |               _/      ← Class 1 predicted here
 0.5 |- - - - - - -_/- - - - - - - - - - -   (decision boundary)
     |          __/          ← Class 0 predicted here
     |     ____/
   0 |____/
     |________________________________ z
```

💡 **Intuition**: With two features, the decision boundary becomes a **line** separating the two classes in feature space. With more features, it becomes a **hyperplane** — directly analogous to how Multiple Linear Regression's line becomes a hyperplane (Chapter 3).

```
Feature 2
    |    o   o
    |  o    o   o
    |______________  ← decision boundary
    |    x   x
    |  x    x   x
    |________________________ Feature 1
    (o = Class 0 region, x = Class 1 region)
```

⭐ **Must Know**: **The 0.5 threshold isn't mandatory** — it's simply the default. In practice, this threshold is often adjusted based on the business problem (e.g., in fraud detection, you might lower the threshold to catch more fraud cases, accepting more false alarms). This connects directly to the next chapter, **Classification Metrics**, where precision/recall tradeoffs formalize this decision.

---

## 9. Binary Classification

This is the default, most common setup: exactly **two classes** (0 and 1, Yes/No, Spam/Not Spam). Everything covered above — Sigmoid, decision boundary, log-odds — is framed around binary classification.

---

## 10. Multi-Class Classification (High Level)

💡 **Intuition**: Real-world problems sometimes have more than two categories (e.g., classifying an email as Personal / Work / Promotions / Spam). Logistic Regression extends to this using two common strategies:

| Strategy | How It Works |
|---|---|
| **One-vs-Rest (OvR)** | Trains one binary Logistic Regression model per class, each asking "is it this class, or not?" — picks the class with the highest predicted probability |
| **Softmax (Multinomial) Regression** | A generalization of the Sigmoid function that directly outputs a probability distribution across all classes simultaneously (probabilities sum to 1) |

📌 **Revision Point**: We're not going deep into multi-class mechanics here — just know these two approaches exist, and that scikit-learn handles multi-class Logistic Regression automatically without you needing to manually implement either strategy.

---

## 11. Training Intuition

💡 **Intuition**: Just like Linear Regression (Chapter 3), Logistic Regression needs a cost function to measure "how wrong" its predictions are, and an optimization process to minimize that cost.

⭐ **Must Know**: Logistic Regression does **not** use MSE (Chapter 4) as its cost function. Instead, it uses **Log Loss** (also called Binary Cross-Entropy) — a cost function specifically designed for probability outputs, which heavily penalizes confident-but-wrong predictions (e.g., predicting 0.99 probability for the class that turns out to be wrong is penalized much more severely than predicting 0.6).

We're not deriving Log Loss here — just know it exists as the probability-appropriate counterpart to MSE, and that Gradient Descent (Chapter 3's high-level concept) is used the same way to minimize it.

📌 **Revision Point**: Same overall training pattern as Linear Regression — a cost function + an optimizer — just a different cost function suited to probability outputs instead of continuous values.

---

## 12. Advantages

⭐ **Must Know** — Logistic Regression works well when:

- The relationship between features and the log-odds of the outcome is roughly linear
- You need **interpretable** results — coefficients directly indicate direction and relative influence on the outcome
- You need genuine **probability outputs**, not just class labels (useful for ranking, risk scoring, thresholding decisions)
- You want a fast, computationally cheap baseline before trying more complex classifiers
- The dataset is small-to-moderate in size

---

## 13. Limitations

| Limitation | Why It Matters |
|---|---|
| **Assumes a roughly linear decision boundary** | Struggles with data that requires a curved or complex boundary to separate classes |
| **Sensitive to outliers** | Extreme feature values can distort the fitted coefficients |
| **Struggles with strongly correlated features** | Similar multicollinearity concerns as Linear Regression (Chapter 5) |
| **Doesn't automatically capture feature interactions** | Needs manually engineered interaction terms to model relationships between combined features |
| **Assumes independent observations** | Same independence assumption from Chapter 5 applies here too |

⚠ **Common Mistake**: Expecting Logistic Regression to perform well on data with clearly non-linear class separation. Just like Linear Regression needed Polynomial Regression (Chapter 6) for curved relationships, Logistic Regression struggles when classes can't be separated by a straight line/hyperplane — this is where tree-based methods (later chapters) often outperform it.

---

## 14. Practical Applications

| Use Case | Why Logistic Regression Fits |
|---|---|
| **Spam detection** | Binary classification, interpretable, fast to deploy at scale |
| **Credit approval / loan default prediction** | Interpretability matters for regulatory and fairness reasons |
| **Medical diagnosis (disease present/absent)** | Probability output is directly useful for risk assessment, not just a hard yes/no |
| **Customer churn prediction** | Probability score can be used to prioritize high-risk customers for retention efforts |
| **Click-through rate estimation** | Requires calibrated probability outputs, not just labels |

---

## 15. sklearn Workflow (High Level)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)          # learns coefficients by minimizing Log Loss
predictions = model.predict(X_test)         # returns class labels (0 or 1)
probabilities = model.predict_proba(X_test) # returns actual probabilities
```

| Step | What It Does |
|---|---|
| `LogisticRegression()` | Creates an untrained classification model |
| `.fit(X_train, y_train)` | Training — learns coefficients that minimize Log Loss |
| `.predict(X_test)` | Returns hard class predictions (using the default 0.5 threshold) |
| `.predict_proba(X_test)` | Returns the underlying probability for each class — useful when you want to apply a custom threshold |

⭐ **Must Know**: `.predict_proba()` is often more useful than `.predict()` in real projects, because it lets you control the decision threshold yourself rather than being locked into the default 0.5 — a concept that becomes very important in the next chapter on Classification Metrics.

---

## 16. Common Beginner Mistakes

| Mistake | Consequence |
|---|---|
| **Using Linear Regression for a classification problem** | Produces meaningless, unbounded outputs with no valid probability interpretation |
| **Interpreting coefficients like Linear Regression's** | Misleading — Logistic Regression coefficients affect log-odds, not the outcome directly |
| **Always using the default 0.5 threshold** | May be inappropriate for the business problem (e.g., fraud, disease detection) — threshold should be a deliberate choice, not a default |
| **Not scaling features** | Can slow convergence and make coefficients harder to interpret, similar to Linear Regression concerns |
| **Assuming accuracy alone tells the full story** | A model can have high accuracy but perform poorly on the class that actually matters — fully addressed in the next chapter |
| **Expecting a linear decision boundary to work on clearly non-linear data** | Underfits — same core issue as using plain Linear Regression on curved data |

---

## 17. Interview Tips

**Q: Why can't you use Linear Regression for classification?**
> Linear Regression's output is unbounded and doesn't represent a valid probability — it can produce values below 0 or above 1, which have no meaningful interpretation as a class decision.

**Q: What does the Sigmoid function do, and why is it used?**
> It transforms the unbounded output of a linear equation into a value between 0 and 1, making it interpretable as a probability. This is what turns a linear model into a valid classifier.

**Q: What is Logistic Regression actually modeling?**
> It models the log-odds of the positive class as a linear combination of the input features. The Sigmoid function then converts that log-odds value back into a probability.

**Q: How do you interpret a coefficient in Logistic Regression?**
> A positive coefficient means an increase in that feature increases the log-odds (and therefore the probability) of the positive class; a negative coefficient decreases it. Unlike Linear Regression, the coefficient doesn't directly translate to a fixed change in the outcome itself.

**Q: What cost function does Logistic Regression use, and why not MSE?**
> It uses Log Loss (Binary Cross-Entropy), which is designed for probability outputs and heavily penalizes confident, incorrect predictions. MSE is designed for continuous targets and isn't well-suited to probability-based classification.

**Q: Is the 0.5 threshold always the right choice?**
> No — the threshold is a business decision. For problems like fraud or disease detection, you might lower the threshold to catch more positive cases, accepting more false positives in exchange for fewer missed true positives.

**Q: How does Logistic Regression handle more than two classes?**
> Through strategies like One-vs-Rest, which trains one binary classifier per class, or Softmax (multinomial) Regression, which directly outputs a probability distribution across all classes at once.

**Q: What's a key limitation of Logistic Regression?**
> It assumes a roughly linear decision boundary between classes, so it struggles when the true separation between classes is non-linear or complex.

---

# Quick Revision

## Equation Summary

```
z (log-odds) = w1*x1 + w2*x2 + ... + wn*xn + b

P(Class 1) = Sigmoid(z) = 1 / (1 + e^(-z))

Decision Rule (default): P ≥ 0.5 → Class 1,  P < 0.5 → Class 0
```

## Terminology Recap

| Term | Meaning |
|---|---|
| Probability | Likelihood of an event, between 0 and 1 |
| Odds | P(event) / (1 - P(event)) |
| Log-Odds (Logit) | log(Odds) — spans all real numbers, matches the linear equation's range |
| Sigmoid Function | Converts any real number back into a probability between 0 and 1 |
| Decision Boundary | The threshold (default 0.5) separating predicted classes |
| Log Loss | The cost function Logistic Regression minimizes during training |
| One-vs-Rest / Softmax | Strategies for extending Logistic Regression to multi-class problems |

## Workflow Recap

```
Prepare Data (Chapter 2)
      ↓
Split into Train/Test
      ↓
model = LogisticRegression()
      ↓
model.fit(X_train, y_train)         → learns coefficients (minimizes Log Loss)
      ↓
model.predict(X_test)               → class labels
model.predict_proba(X_test)         → probabilities
      ↓
Evaluate (next chapter: Classification Metrics)
```

## Interview Facts Cheat Sheet

- Logistic Regression is a classification algorithm, despite the name.
- It models log-odds as a linear function of features, then applies Sigmoid to get a probability.
- Coefficients affect log-odds, not the outcome directly — interpret sign more than raw magnitude.
- Uses Log Loss, not MSE, as its cost function.
- The 0.5 decision threshold is a default, not a rule — adjust based on business needs.
- Struggles with non-linear decision boundaries, similar to how Linear Regression struggles with curved data.
- `.predict_proba()` gives you the raw probability; `.predict()` applies the default threshold.

## ✅ Checklist: "I understand this chapter if I can..."

- [ ] Explain why Linear Regression fails for classification problems
- [ ] Explain probability, odds, and log-odds, and how they connect
- [ ] Describe the Sigmoid function's shape and purpose
- [ ] Write and explain the Logistic Regression equation
- [ ] Explain what a decision boundary is and why 0.5 isn't mandatory
- [ ] Describe the difference between binary and multi-class classification approaches
- [ ] Explain why Logistic Regression uses Log Loss instead of MSE
- [ ] List at least 3 strengths and 3 limitations of Logistic Regression
- [ ] Explain the difference between `.predict()` and `.predict_proba()`
- [ ] Answer every interview question in this chapter without looking