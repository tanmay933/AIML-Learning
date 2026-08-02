# Imbalanced Data

## Learning Objectives

By the end of this chapter, you will be able to:

- Define imbalanced data and recognize it in real-world datasets
- Explain why accuracy fails as a metric on imbalanced problems
- Distinguish minority and majority classes and understand their impact on training
- Understand under-sampling, over-sampling, and SMOTE at a practical level
- Understand class weighting and threshold adjustment as alternatives to resampling
- Choose appropriate evaluation metrics for imbalanced problems
- Follow a correct, leakage-free workflow for handling imbalanced data
- Answer interview questions on imbalanced data with engineering-grade clarity

---

# Why This Topic Exists

Chapter 11 ended with a warning that echoes throughout this chapter: a fraud detection model that always predicts "not fraud" can still hit 99% accuracy while catching zero actual fraud. That's not a hypothetical edge case — it's the **default failure mode** of nearly every real-world classification problem, because most real-world classification problems are imbalanced.

Fraud, disease, churn, defects, security breaches — the event you actually care about detecting is almost always the rare one. This chapter teaches you how to recognize imbalance, understand why it silently sabotages standard training and evaluation, and apply the engineering techniques used to fix it.

---

# Intuition

## What Is Imbalanced Data?

💡 **Intuition**: Imbalanced data is when one class vastly outnumbers another in your dataset — for example, 99% "not fraud" and 1% "fraud."

```
Balanced dataset:              Imbalanced dataset:
Class A: 50%                   Class A: 99%
Class B: 50%                   Class B: 1%

  A A A A A                      A A A A A A A A A A
  B B B B B                      A A A A A A A A A A
                                 B
```

⭐ **Must Know**: Imbalance isn't just a data quirk — it's the **natural state of most interesting classification problems.** Rare, high-stakes events (fraud, disease, failure) are rare *by definition* — that's precisely why detecting them matters.

## Why Accuracy Fails

📌 **Revision Point**: This was introduced in Chapter 11 — a model that predicts the majority class 100% of the time still achieves accuracy roughly equal to the majority class's proportion, while providing **zero real value**. This chapter is dedicated to fixing the root cause behind that failure mode, not just the metric used to detect it.

```
99% "Not Fraud", 1% "Fraud"

"Lazy model": always predicts "Not Fraud"
→ Accuracy = 99%
→ Recall on Fraud class = 0%   (catches nothing)
```

⚠ **Common Mistake**: Believing a high-accuracy model is automatically a good model. On imbalanced data, this assumption is actively dangerous — always check class-specific performance (Precision, Recall, F1 — Chapter 11) before trusting accuracy.

---

# Core Concepts

## 1. Business Examples

| Domain | Majority Class | Minority Class (the one that matters) | Typical Imbalance |
|---|---|---|---|
| Fraud detection | Legitimate transactions | Fraudulent transactions | 99%+ / <1% |
| Disease screening | Healthy patients | Patients with the disease | Often 95%+ / <5% |
| Manufacturing defect detection | Non-defective items | Defective items | 98%+ / <2% |
| Customer churn | Retained customers | Churned customers | Varies, often 80-90% / 10-20% |
| Spam detection | Legitimate email | Spam | Varies widely by dataset |

💡 **Intuition**: In every one of these cases, the **minority class is the actual point of the model.** Nobody builds a fraud detector to confirm that most transactions are fine — they build it to catch the rare bad ones.

---

## 2. Minority vs Majority Classes

| Term | Meaning |
|---|---|
| **Majority Class** | The class with far more examples in the dataset |
| **Minority Class** | The class with far fewer examples — usually the class of actual business interest |

⭐ **Must Know**: **Standard training treats every data point equally**, which means a model trained on imbalanced data has strong statistical incentive to focus on the majority class — it can achieve low overall error just by being lazy about the minority class, since there are so few minority examples to get "wrong" in the first place.

💡 **Intuition**: Imagine studying for an exam where 99% of questions are about Topic A and 1% are about Topic B. If you only study Topic A, you'll still score ~99% — the model, in a sense, "learns" the same lazy strategy unless it's specifically corrected for it.

---

## 3. Under-Sampling

**How it works**: Remove examples from the **majority class** until the classes are more balanced.

```
Before:  [Majority: 9000] [Minority: 1000]
After:   [Majority: 1000] [Minority: 1000]
              ↑ reduced to match minority
```

| Advantages | Disadvantages |
|---|---|
| Faster training (smaller dataset) | Discards potentially useful data |
| Simple to implement | Can lose important patterns present only in the removed majority examples |
| Reduces the majority class's dominance | Risk of underfitting if too much data is removed |

⚠ **Common Mistake**: Under-sampling too aggressively on an already-small dataset, leaving too little data overall to learn any reliable pattern.

---

## 4. Over-Sampling

**How it works**: Duplicate (or otherwise increase) examples from the **minority class** until the classes are more balanced.

```
Before:  [Majority: 9000] [Minority: 1000]
After:   [Majority: 9000] [Minority: 9000]
                                ↑ increased to match majority
```

| Advantages | Disadvantages |
|---|---|
| No data is discarded | Simple duplication can cause overfitting — the model may memorize repeated exact copies |
| Preserves all majority-class information | Increases dataset size and training time |
| Simple to implement | Doesn't add genuinely new information, just repeats existing points |

⚠ **Common Mistake**: Naive over-sampling (exact duplication) can make a model overconfident about a small number of specific minority examples, rather than learning the general pattern behind the minority class — a subtle form of overfitting (Chapter 7).

---

## 5. SMOTE (High Level)

**SMOTE** = Synthetic Minority Over-sampling Technique.

💡 **Intuition**: Instead of just duplicating existing minority examples (which can cause overfitting), SMOTE creates **new, synthetic minority examples** by interpolating between existing minority data points — essentially generating plausible "in-between" examples rather than exact copies.

```
Existing minority points:        SMOTE-generated synthetic points:
    •         •                       •    ×    •
                                            ↑
         •                          new synthetic point created
                                     "between" nearby minority points
```

⭐ **Must Know**: SMOTE reduces the overfitting risk of naive duplication because the model sees **varied, synthetic examples** rather than identical repeats — it's forced to learn a more general boundary around the minority class rather than memorizing specific points.

**When to use it**: Generally preferred over simple over-sampling when you have enough minority examples to interpolate between meaningfully. Less reliable with extremely few minority examples (too little signal to interpolate from).

We're not deriving the interpolation mechanics here — just understand the core idea: **synthetic, varied examples beat exact duplicates.**

---

## 6. Class Weights

💡 **Intuition**: Instead of changing the *data* (like under/over-sampling), you can change how the *model's cost function* treats mistakes — telling the model "an error on the minority class costs more than an error on the majority class."

```
Standard cost function:      Every mistake weighted equally

Weighted cost function:      Minority-class mistakes weighted more heavily
                              (e.g., 10x penalty for missing a fraud case
                               vs. a false alarm on a legit transaction)
```

⭐ **Must Know**: This directly modifies training — connecting back to the cost function concept from Chapters 3 and 10 (MSE, Log Loss). A weighted cost function penalizes the model more for getting minority-class predictions wrong, pushing it to pay closer attention to that class during optimization.

**Advantages over resampling:**
- No changes to the dataset itself — no data duplicated or discarded
- Often simpler to implement (many libraries support a `class_weight` parameter directly)
- Avoids the overfitting risk of naive over-sampling

🎯 **Interview Tip**: If asked "How would you handle imbalanced data without changing the dataset?" — class weighting is the answer: it adjusts the training process itself, not the data.

---

## 7. Threshold Adjustment (High Level)

📌 **Revision Point**: Recall from Chapter 10 — Logistic Regression's default 0.5 decision threshold is just a convention, not a rule.

💡 **Intuition**: On imbalanced data, the default threshold often isn't appropriate — a model might rarely output a probability above 0.5 for the rare minority class, even when it has genuinely useful information. **Lowering the threshold** (e.g., to 0.3 or 0.2) makes the model predict the minority class more often, trading some precision for higher recall.

```
Default threshold (0.5):        Lowered threshold (0.3):
Fewer minority predictions       More minority predictions
Higher precision, lower recall   Lower precision, higher recall
```

⭐ **Must Know**: Threshold adjustment is one of the **cheapest and most flexible** techniques available — it requires no retraining, no changes to the dataset, just choosing a different cutoff on the same trained model's output probabilities. This connects directly to the Precision-Recall tradeoff from Chapter 11.

---

## 8. Choosing Evaluation Metrics

📌 **Revision Point**: This was previewed in Chapter 11 — imbalanced data is precisely the scenario where metric choice matters most.

| Metric | Suitable for Imbalanced Data? | Why |
|---|---|---|
| Accuracy | ❌ No | Dominated by the majority class; can hide complete failure on the minority class |
| Precision | ✅ Yes | Focuses on quality of positive predictions |
| Recall | ✅ Yes | Focuses on coverage of the minority (positive) class — usually the priority |
| F1 Score | ✅ Yes | Balances Precision and Recall, doesn't get inflated by majority-class performance |
| ROC-AUC | ⚠ Use with caution | Can look overly optimistic since FPR is calculated against a large number of true negatives |
| Precision-Recall AUC | ✅ Yes, often preferred | Focuses entirely on the positive/minority class, unaffected by majority-class volume |

⭐ **Must Know**: **Never rely on accuracy for imbalanced problems.** Use Precision, Recall, F1, and Precision-Recall curves instead — all covered in depth in Chapter 11, now applied specifically to this scenario.

---

## 9. Practical Workflow

```
Collect Data
      ↓
Check Class Distribution           ← is the data actually imbalanced?
      ↓
Split into Train/Test              ← BEFORE any resampling (see below)
      ↓
Apply resampling (SMOTE/under/over-sampling)
   ONLY on the training set
      ↓
OR apply class weights during model training
(instead of resampling — often simpler and safer)
      ↓
Train Model
      ↓
Evaluate on the ORIGINAL, untouched, imbalanced test set
   using Precision, Recall, F1, PR-AUC
      ↓
Adjust decision threshold if needed based on business priorities
```

⭐ **Must Know**: **Resampling must be applied only to the training set — never to the test set.** The test set must reflect the real-world class distribution the model will actually face in production. This is a direct extension of the data leakage principle from Chapter 2 and the Cross Validation workflow from Chapter 9: any transformation of the data must respect the train/test boundary.

⚠ **Common Mistake**: Resampling (or applying SMOTE) on the **entire dataset before splitting** — this leaks synthetic or duplicated information across the train/test boundary and produces a test set that no longer reflects reality, making evaluation results meaningless.

---

## 10. Common Beginner Mistakes

| Mistake | Consequence |
|---|---|
| **Trusting accuracy on imbalanced data** | Model looks great on paper, fails completely at its actual purpose |
| **Resampling before splitting into train/test** | Data leakage — evaluation no longer reflects real-world performance |
| **Resampling the test set** | Test set no longer represents the true class distribution the model will face in production |
| **Naive over-sampling (duplication) without considering overfitting risk** | Model may memorize repeated minority examples instead of learning general patterns |
| **Over-sampling too aggressively** | Can push the model to over-predict the minority class, hurting precision significantly |
| **Ignoring threshold adjustment as a lower-cost alternative** | Missing a simple, effective fix that doesn't require retraining or resampling |
| **Assuming one technique (e.g., SMOTE) is always the best choice** | Different techniques suit different situations — dataset size, degree of imbalance, and algorithm choice all matter |

---

## 11. Interview Tips

**Q: What is imbalanced data, and why is it common in real-world ML?**
> Imbalanced data occurs when one class vastly outnumbers another. It's common because the events businesses most want to detect — fraud, disease, defects, churn — are inherently rare by nature.

**Q: Why does accuracy fail on imbalanced datasets?**
> A model can achieve high accuracy simply by always predicting the majority class, while completely failing to identify the minority class — which is usually the class of actual interest.

**Q: What's the difference between under-sampling and over-sampling?**
> Under-sampling removes examples from the majority class to balance the dataset, which risks discarding useful data. Over-sampling duplicates or generates new examples for the minority class, which risks overfitting if done naively through simple duplication.

**Q: What is SMOTE, and why is it often preferred over simple over-sampling?**
> SMOTE generates synthetic minority-class examples by interpolating between existing minority points, rather than exactly duplicating them. This reduces overfitting risk compared to naive duplication, since the model sees varied examples instead of identical repeats.

**Q: How do class weights help with imbalanced data?**
> Class weights modify the model's cost function to penalize mistakes on the minority class more heavily, without changing the underlying dataset — encouraging the model to pay more attention to the minority class during training.

**Q: Why should resampling only be applied to the training set?**
> The test set must reflect the real-world class distribution the model will face in production. Resampling the test set (or resampling before splitting) leaks information and produces an evaluation that doesn't represent actual real-world performance.

**Q: What's a lower-cost alternative to resampling for handling imbalance?**
> Adjusting the decision threshold on the model's output probabilities — this requires no retraining or data changes, and directly trades off precision and recall to suit the business need.

**Q: Which evaluation metrics should you avoid, and which should you use, for imbalanced data?**
> Avoid relying on accuracy, since it's dominated by the majority class. Use Precision, Recall, F1 Score, and Precision-Recall curves instead, since they focus on the performance that actually matters for the minority class.

---

# Quick Revision

## Techniques Summary

| Technique | What It Changes | Key Risk |
|---|---|---|
| Under-sampling | Removes majority-class examples | Loses potentially useful data |
| Over-sampling | Duplicates minority-class examples | Overfitting to repeated examples |
| SMOTE | Generates synthetic minority examples | Less reliable with very few minority samples |
| Class Weights | Modifies the cost function, not the data | Requires algorithm support (`class_weight` parameter) |
| Threshold Adjustment | Changes the decision cutoff on probabilities | Shifts precision/recall tradeoff, doesn't fix underlying training bias |

## Terminology Recap

| Term | Meaning |
|---|---|
| Imbalanced Data | Dataset where one class significantly outnumbers another |
| Majority Class | The over-represented class |
| Minority Class | The under-represented class, usually the class of interest |
| Under-sampling | Reducing majority-class examples |
| Over-sampling | Increasing minority-class examples (via duplication) |
| SMOTE | Synthetic Minority Over-sampling Technique — generates new synthetic minority examples |
| Class Weight | A training-time penalty adjustment favoring the minority class |

## Correct Workflow Recap

```
Check class distribution → Split Train/Test → Resample TRAINING SET ONLY
   (or use class weights instead) → Train Model
   → Evaluate on ORIGINAL test set using Precision/Recall/F1/PR-AUC
   → Adjust threshold if needed
```

## Interview Facts Cheat Sheet

- Imbalance is the default state of most real, high-value classification problems.
- Accuracy is unreliable on imbalanced data — always check Precision/Recall/F1.
- Under-sampling loses data; over-sampling risks overfitting via duplication.
- SMOTE creates synthetic examples instead of exact duplicates — generally safer than naive over-sampling.
- Class weights modify training itself, without touching the dataset.
- Threshold adjustment is the cheapest fix — no retraining or resampling required.
- Resampling must happen only on the training set, never the test set — same leakage principle as Chapter 2 and Chapter 9.
- Precision-Recall curves are generally preferred over ROC-AUC for heavily imbalanced problems.

## ✅ Checklist: "I understand this chapter if I can..."

- [ ] Define imbalanced data and give 3 real-world examples
- [ ] Explain precisely why accuracy fails on imbalanced datasets
- [ ] Compare under-sampling, over-sampling, and SMOTE, including their tradeoffs
- [ ] Explain how class weights change training without touching the dataset
- [ ] Explain how threshold adjustment works and why it's a low-cost fix
- [ ] List which evaluation metrics are appropriate (and inappropriate) for imbalanced data
- [ ] Describe the correct, leakage-free workflow for handling imbalanced data
- [ ] Explain why resampling must never touch the test set
- [ ] Answer every interview question in this chapter without looking