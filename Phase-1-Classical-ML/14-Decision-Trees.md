# Decision Trees

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain why Decision Trees exist and how they differ fundamentally from every algorithm covered so far
- Understand tree structure: root, internal nodes, leaves, and branches
- Explain recursive splitting and how a tree is actually built
- Understand impurity, Gini Impurity, Entropy, and Information Gain
- Explain stopping criteria and pruning at a practical level
- Understand how Decision Trees handle regression, not just classification
- Explain feature importance and how trees handle missing values
- Identify Decision Trees' strengths, limitations, and complexity characteristics
- Answer interview questions on Decision Trees with engineering-grade clarity
- Be fully prepared to understand Random Forest and Boosting in the next chapters

---

# Why This Topic Exists

Every algorithm so far has made decisions using **distance** (KNN) or a **weighted sum of features** (Linear/Logistic Regression). Decision Trees introduce a completely different, arguably more human, way of making predictions: **ask a series of yes/no questions about the data until you arrive at an answer.**

This chapter matters enormously for what comes next. **Random Forest** and **Boosting** — two of the most powerful and widely used algorithms in real-world ML — are both built directly on top of Decision Trees. Understanding a single tree deeply is the prerequisite for understanding why combining many trees works so well.

---

# Intuition

💡 **Intuition**: Think about how a doctor might informally diagnose a patient:

```
Is the patient's temperature > 100.4°F?
        /                    \
      Yes                     No
       |                       |
Is there a cough?        Probably not sick
   /        \
 Yes         No
  |           |
Likely flu   Possibly a
             different issue
```

This is exactly how a Decision Tree works — a sequence of simple, interpretable yes/no questions, each one narrowing down the possibilities, until you reach a final decision.

⭐ **Must Know**: Decision Trees are one of the few ML algorithms that are **directly human-readable** — you can literally trace the exact reasoning behind any single prediction, question by question. This interpretability is a major reason they're popular in regulated industries (finance, healthcare, insurance).

---

# Core Concepts

## 1. Tree Structure

```
                    [Root Node]
                   Area > 1500?
                  /            \
               Yes               No
                /                  \
        [Internal Node]         [Leaf Node]
        Bedrooms > 3?          Predict: $180K
        /            \
      Yes              No
       |                |
  [Leaf Node]      [Leaf Node]
  Predict: $350K   Predict: $250K
```

| Term | Meaning |
|---|---|
| **Root Node** | The very first question — splits the entire dataset |
| **Internal Node** | A subsequent question, splitting an already-narrowed subset of the data |
| **Leaf Node** | The end of a path — no more questions, this is the final prediction |
| **Branch** | The connection between nodes, representing the answer (Yes/No, or a threshold comparison) to a question |

📌 **Revision Point**: Every path from the root to a leaf represents one complete "rule" — e.g., "if Area > 1500 AND Bedrooms > 3, predict $350K."

---

## 2. Recursive Splitting

💡 **Intuition**: A tree is built by repeatedly asking: **"What single question, asked right now, best separates this data into cleaner, more homogeneous groups?"** — then applying that same question-asking process again to each resulting group, and again, and again.

```
Full Dataset
      ↓ (find best split)
   Split into two groups
      ↓                    ↓
(find best split)      (find best split)
      ↓                    ↓
  Split again           Split again
      ↓                    ↓
    ...                  ...
```

⭐ **Must Know**: This is called **recursive** splitting because the exact same "find the best split" procedure is applied over and over, to smaller and smaller subsets of data, until some stopping condition is reached (Section 10).

---

## 3. Decision Boundaries

💡 **Intuition**: Unlike Logistic Regression's smooth, linear decision boundary (Chapter 10), a Decision Tree's boundary is made of **axis-aligned rectangular splits** — because each question only looks at one feature at a time, compared against a threshold.

```
Feature 2
    |    o   o  |  x   x
    |  o    o   |    x
    |___________|________
    |    o   o  |  x   x
    |  o    o   |    x
    |________________________ Feature 1
         (boundary is a straight vertical/horizontal cut,
          not a diagonal line like Logistic Regression)
```

⭐ **Must Know**: Because splits are always **axis-aligned** (one feature, one threshold, at a time), a tree can approximate *any* shape of decision boundary given enough splits — including boundaries that Logistic Regression's straight line could never represent. This is one of the biggest structural differences from every algorithm in Chapters 3–13.

---

## 4. Impurity

**What it means**: A measure of how "mixed" the classes are within a node. A node with only one class present is **pure**; a node with an even mix of classes is **maximally impure**.

```
Pure node:               Impure node:
  o o o o                  o x o x
  o o o o                  x o x o
  (100% Class o)           (50/50 mix)
```

💡 **Intuition**: Every split a tree considers is judged by one question: **"Does this split make the resulting groups purer (more homogeneous) than before?"** The tree always picks the split that improves purity the most.

---

## 5. Gini Impurity

💡 **Intuition**: Measures the probability that a randomly picked item from a node would be **misclassified** if you randomly labeled it according to the class distribution in that node.

```
Gini = 1 − (probability of Class A)² − (probability of Class B)² − ...
```

| Node Composition | Gini Impurity |
|---|---|
| 100% one class (pure) | 0 (minimum — perfectly pure) |
| 50/50 split between two classes | 0.5 (maximum impurity for 2 classes) |

⭐ **Must Know**: **Lower Gini = purer node = better split.** A Gini of 0 means the node contains only one class — no further splitting needed there. We're not deriving this formula — just understand it as a number between 0 and 0.5 (for binary classification) that measures mixedness.

---

## 6. Entropy

💡 **Intuition**: Borrowed from information theory — measures the amount of "disorder" or "uncertainty" in a node's class distribution. Conceptually very similar to Gini, just calculated differently (using logarithms, which we won't derive here).

```
Entropy = 0    → node is perfectly pure (no uncertainty)
Entropy = 1    → maximum uncertainty (50/50 split, binary classification)
```

📌 **Revision Point**: Gini and Entropy almost always lead to very similar tree structures in practice. Gini is slightly faster to compute (no logarithms), so it's the default in many libraries, including scikit-learn.

| Aspect | Gini Impurity | Entropy |
|---|---|---|
| Range (binary) | 0 to 0.5 | 0 to 1 |
| Computation | Faster (no logarithm) | Slightly slower |
| Typical results | Very similar to Entropy in practice | Very similar to Gini in practice |

🎯 **Interview Tip**: If asked "Gini vs Entropy — does it matter which one you use?" — answer: *"In practice, they usually produce very similar trees. Gini is computationally cheaper since it avoids logarithms, which is why it's the common default."*

---

## 7. Information Gain

💡 **Intuition**: **Information Gain** measures how much a particular split **reduces impurity** (using either Gini or Entropy) compared to before the split.

```
Information Gain = Impurity(before split) − Weighted Average Impurity(after split)
```

⭐ **Must Know**: At every node, a Decision Tree evaluates **every possible split** (every feature, every possible threshold) and picks the one with the **highest Information Gain** — the split that produces the purest possible resulting groups.

```
Before split:  Impurity = 0.5 (very mixed)
                    ↓
        Split on "Area > 1500?"
                    ↓
After split:   Left group Impurity = 0.1
               Right group Impurity = 0.15
                    ↓
Information Gain = 0.5 − (weighted average of 0.1 and 0.15) = large gain
                    → this is a GOOD split
```

🎯 **Interview Tip**: If asked "How does a Decision Tree decide what to split on?" — answer: *"At each node, it evaluates all possible splits across all features and picks the one that maximizes Information Gain — i.e., the split that most reduces impurity (Gini or Entropy) in the resulting child nodes."*

---

## 8. Building the Tree — Putting It Together

```
1. Start with the full dataset at the root node.
2. For every feature, and every possible threshold, calculate Information Gain.
3. Pick the split with the highest Information Gain.
4. Create two child nodes based on that split.
5. Repeat steps 2–4 recursively on each child node.
6. Stop when a stopping criterion is met (Section 10).
```

💡 **Intuition**: This is a **greedy** algorithm — at every step, it picks the *locally* best split available right now, without looking ahead to see if a different split might lead to better splits later. This greedy nature is a key limitation, discussed further in Section 14.

---

## 9. Stopping Criteria

⭐ **Must Know**: Without a stopping rule, a tree would keep splitting until every leaf is **perfectly pure** — often meaning just one data point per leaf. This is a direct path to severe overfitting (Chapter 7).

Common stopping criteria:

| Criterion | What It Does |
|---|---|
| **Max Depth** | Limits how many levels of questions the tree can ask |
| **Min Samples per Leaf** | Requires a minimum number of data points in each leaf (prevents overly specific, tiny leaves) |
| **Min Samples to Split** | Requires a minimum number of data points in a node before it's allowed to split further |
| **Max Leaf Nodes** | Caps the total number of leaves in the tree |
| **Minimum Impurity Decrease** | Only allows a split if it improves purity by at least a certain amount |

```
No stopping criteria (grows until pure):     With max_depth = 3:

        [root]                                     [root]
       /      \                                   /      \
    [node]   [node]                            [node]   [node]
    /   \    /   \                              /  \      /  \
  ... deep tree ...                          [leaf][leaf][leaf][leaf]
  (each leaf = 1 point,                      (stops at depth 3,
   severely overfit)                          generalizes better)
```

⭐ **Must Know**: This is the **direct control knob for the Bias-Variance Tradeoff (Chapter 7)** in Decision Trees — a shallow tree (low max depth) has higher bias; a deep, unconstrained tree has higher variance. Tuning these stopping criteria is exactly analogous to choosing polynomial degree (Chapter 6) or K in KNN (Chapter 13).

---

## 10. Pruning (High Level)

💡 **Intuition**: Instead of stopping the tree early (Section 9), **pruning** takes the opposite approach: let the tree grow fully, then **cut back** branches afterward that don't meaningfully improve performance on validation data.

```
Fully grown tree                    Pruned tree
    (overfit)                    (branches removed that
        |                          didn't help generalization)
   [deep, complex]        →              [simpler tree]
```

📌 **Revision Point**: Stopping criteria prevent overfitting **proactively** (during growth); pruning fixes overfitting **retroactively** (after growth). Both aim at the same goal: a tree that generalizes well rather than memorizing training data. We're not covering pruning algorithms in depth here — just know the concept and its purpose.

---

## 11. Regression Trees

💡 **Intuition**: Decision Trees aren't limited to classification. For regression, instead of splitting to maximize class purity (Gini/Entropy), the tree splits to **minimize variance** of the target values within each resulting group — and each leaf predicts the **average** target value of the samples that land there.

```
Regression tree leaf:
   Samples in leaf: [$300K, $320K, $310K, $290K]
   Leaf prediction = average = $305K
```

⭐ **Must Know**: The overall mechanism (recursive splitting, greedy best-split selection) is identical to classification trees — only the **splitting criterion** (variance reduction instead of Gini/Entropy) and the **leaf prediction** (average instead of majority vote) change.

📌 **Revision Point**: This mirrors exactly what you saw with KNN in Chapter 13 — same core algorithm, different aggregation depending on whether the task is classification (vote/majority-class) or regression (average).

---

## 12. Advantages

⭐ **Must Know** — Decision Trees work well when:

- **Interpretability matters** — the entire decision path can be traced and explained to non-technical stakeholders
- The relationship between features and target is **non-linear or has complex interactions** (Chapter 10 mentioned Logistic Regression struggles here — trees don't)
- **No feature scaling is required** (Chapter 2/3 callback) — since splits are based on threshold comparisons per feature, not distances or magnitudes
- The data has a **mix of numerical and categorical features** — trees handle both naturally without needing extensive encoding
- You need a strong **foundation for ensemble methods** (Random Forest, Boosting — next chapters)

---

## 13. Limitations

| Limitation | Why It Matters |
|---|---|
| **Prone to overfitting** | Without constraints, trees grow until perfectly (over)fit to training data |
| **Greedy, not globally optimal** | Picks the locally best split at each step; may miss a better overall tree structure |
| **Unstable** | Small changes in training data can produce a very different tree structure — a form of high variance (Chapter 7) |
| **Biased toward features with many possible split points** | Features with more unique values have more opportunities to appear "useful," even if not genuinely more informative |
| **Poor at extrapolation** | Like KNN, trees can only predict values seen in training data ranges — leaf predictions are bounded by training data |

⚠ **Common Mistake**: Assuming a single, unconstrained Decision Tree will generalize well. In practice, individual trees are **notoriously prone to overfitting** — this instability is precisely the motivation for Random Forest, which averages many trees together to cancel out this instability (previewed here, covered fully next chapter).

---

## 14. Feature Importance

💡 **Intuition**: Because every split is chosen based on how much it improves purity (Information Gain), you can measure **how much each feature contributed, in total, to reducing impurity across the entire tree** — this gives a ranked measure of feature importance.

```
Feature Importance (example):
Area           ████████████████ 45%
Location Score ██████████ 30%
Bedrooms       ███████ 18%
Year Built     ██ 7%
```

⭐ **Must Know**: Feature importance is one of the most practically useful outputs of a Decision Tree — it directly tells you which features the model actually relied on, unlike KNN (Chapter 13), which has no such built-in interpretability signal.

🚀 **Practical Insight**: Feature importance from trees is commonly used as a quick, practical feature selection tool even *outside* of tree-based models — engineers often train a simple tree just to identify which features matter most, then use that insight elsewhere.

---

## 15. Handling Missing Values (High Level)

💡 **Intuition**: Unlike many algorithms (Linear/Logistic Regression, KNN) that require missing values to be handled beforehand (Chapter 2's imputation strategies), some Decision Tree implementations can handle missing values more natively — for example, by learning a "default direction" to send missing values during a split, based on which direction produces better splits on the available data.

📌 **Revision Point**: This capability **varies by implementation** — scikit-learn's standard `DecisionTreeClassifier`/`Regressor` still generally expects missing values to be handled beforehand (Chapter 2 imputation), while some other libraries (e.g., certain gradient boosting implementations, covered later) handle missing values internally. Don't assume every tree implementation handles missing data automatically.

---

## 16. Time Complexity

| Phase | Complexity (approximate) |
|---|---|
| **Training** | O(n × d × log n) — roughly, for `n` samples and `d` features, considering all features and thresholds at each of the O(log n) levels |
| **Prediction** | O(log n) per sample — just traversing from root to leaf, following one path down the tree |

⭐ **Must Know**: **Prediction is very fast** — a huge advantage over KNN (Chapter 13), which requires comparing against the entire training set at prediction time. A trained Decision Tree only needs to follow one path of yes/no questions, roughly proportional to the tree's depth.

---

## 17. Practical Applications

| Use Case | Why Decision Trees Fit |
|---|---|
| **Credit approval decisions** | Interpretability is often legally required — trees produce explainable rules |
| **Medical diagnosis support systems** | Clear decision paths can be reviewed and validated by clinicians |
| **Customer segmentation rules** | Easy to translate into actionable business rules |
| **Fraud detection (as part of ensembles)** | Often used as building blocks within Random Forest/Boosting for stronger performance |
| **Any problem needing an explainable model** | When stakeholders need to understand *why* a prediction was made, not just *what* it is |

---

## 18. sklearn Workflow (High Level)

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,               # controls overfitting (Section 9)
    min_samples_leaf=10,       # controls overfitting (Section 9)
    criterion='gini'           # 'gini' or 'entropy' (Sections 5-6)
)
model.fit(X_train, y_train)    # recursively builds the tree
predictions = model.predict(X_test)
importances = model.feature_importances_   # Section 14
```

| Parameter | Role |
|---|---|
| `max_depth` | Stopping criterion — limits tree depth |
| `min_samples_leaf` | Stopping criterion — minimum samples per leaf |
| `criterion` | Splitting metric — `'gini'` or `'entropy'` |
| `.feature_importances_` | Ranked importance of each feature after training |

⭐ **Must Know**: No feature scaling step is needed here — unlike the KNN workflow in Chapter 13, which required `StandardScaler` before fitting.

---

## 19. Common Beginner Mistakes

| Mistake | Consequence |
|---|---|
| **Not setting max_depth or other stopping criteria** | Tree grows unconstrained, severely overfitting (near-zero training error, poor test error) |
| **Judging tree quality on training accuracy alone** | Same trap as every prior chapter — always check test/validation performance (Chapter 4/7) |
| **Assuming trees need feature scaling** | Unnecessary step — trees are threshold-based, not distance-based |
| **Ignoring tree instability** | Small data changes can produce very different trees — a single tree's structure shouldn't be over-interpreted as "the truth" |
| **Not checking feature importance** | Misses a valuable, free interpretability signal that trees provide naturally |
| **Using very deep trees on small datasets** | Combines the worst of both: overfitting risk and unstable, unreliable splits from limited data |

---

## 20. Interview Tips

**Q: How does a Decision Tree decide what feature and threshold to split on?**
> At each node, it evaluates all possible splits across all features and thresholds, calculates the Information Gain (reduction in Gini Impurity or Entropy) for each, and selects the split that maximizes Information Gain.

**Q: What's the difference between Gini Impurity and Entropy?**
> Both measure how "mixed" the classes are in a node, and both are minimized by pure nodes. Gini is computationally simpler (no logarithms) and is the common default; Entropy is derived from information theory. In practice, they usually produce very similar trees.

**Q: Why do Decision Trees not require feature scaling?**
> Because splits are based on threshold comparisons on individual features (e.g., "Area > 1500?"), which are unaffected by the relative scale of different features — unlike distance-based algorithms like KNN.

**Q: How do you prevent a Decision Tree from overfitting?**
> By using stopping criteria such as max depth, minimum samples per leaf, or minimum samples to split — or by growing the tree fully and then pruning it back afterward based on validation performance.

**Q: What is feature importance in a Decision Tree?**
> A measure of how much each feature contributed, in total, to reducing impurity across all its splits in the tree — giving a ranked view of which features the model relied on most.

**Q: Why is a single Decision Tree considered "unstable"?**
> Small changes in the training data can lead to very different splits being chosen (since it's a greedy algorithm), which can produce a substantially different tree structure — a form of high variance.

**Q: How does a Regression Tree differ from a Classification Tree?**
> A Regression Tree splits to minimize variance within resulting groups (instead of maximizing class purity) and predicts the average target value at each leaf (instead of taking a majority class vote).

**Q: Why are Decision Trees considered a good foundation for Random Forest?**
> Because a single tree's key weakness — instability and overfitting — can be addressed by combining many trees together and averaging their predictions, which is exactly what Random Forest does.

---

# Quick Revision

## Core Concept Summary

```
Recursive Splitting:
  At each node → try every feature/threshold → pick split with highest Information Gain
             → repeat on each resulting child node
             → stop when a stopping criterion is met
             → leaf nodes make the final prediction (majority vote or average)
```

## Terminology Recap

| Term | Meaning |
|---|---|
| Root Node | First split of the full dataset |
| Internal Node | A subsequent split on an already-narrowed subset |
| Leaf Node | End of a path — holds the final prediction |
| Impurity | How mixed the classes are in a node |
| Gini Impurity | A specific impurity measure (0 = pure, up to 0.5 for binary) |
| Entropy | Alternative impurity measure from information theory (0 to 1) |
| Information Gain | Reduction in impurity achieved by a split |
| Pruning | Cutting back a fully grown tree to reduce overfitting |
| Feature Importance | Ranked measure of how much each feature reduced impurity overall |

## Complexity Recap

| Phase | Complexity |
|---|---|
| Training | O(n × d × log n) approx. |
| Prediction | O(log n) per sample — very fast |

## Interview Facts Cheat Sheet

- Trees split recursively, greedily choosing the split with highest Information Gain at each step.
- Lower Gini/Entropy = purer node = better split.
- No feature scaling required — trees use thresholds, not distances.
- Unconstrained trees overfit severely — max depth / min samples / pruning control this.
- A single tree is unstable (high variance) — this instability motivates Random Forest (next chapter).
- Regression trees predict the average of leaf samples; classification trees use majority vote.
- Prediction is fast (O(log n)) — a major advantage over KNN's O(n×d) prediction cost.
- Feature importance is a natural, built-in interpretability tool trees provide.

## ✅ Checklist: "I understand this chapter if I can..."

- [ ] Draw and label a simple Decision Tree (root, internal node, leaf, branch)
- [ ] Explain recursive splitting and the greedy nature of tree-building
- [ ] Explain Gini Impurity and Entropy, and how they relate to Information Gain
- [ ] Explain how a tree selects the best split at each node
- [ ] List at least 3 stopping criteria and explain why they matter
- [ ] Explain the difference between stopping criteria and pruning
- [ ] Explain how Regression Trees differ from Classification Trees
- [ ] Explain why trees don't require feature scaling
- [ ] Explain feature importance and how it's derived
- [ ] Explain why a single tree is unstable, and why that motivates Random Forest
- [ ] Answer every interview question in this chapter without looking