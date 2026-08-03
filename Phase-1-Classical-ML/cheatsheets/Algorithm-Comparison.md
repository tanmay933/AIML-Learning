# Machine Learning Algorithm Comparison Guide

---

# 1. Complete Algorithm Comparison Table

| Algorithm | Type | Task | Labels Needed? | Numerical Data | Categorical Data | Scaling Needed? | Non-Linear? | Outlier Sensitive? | Interpretability | Train Speed | Predict Speed | Memory | Overfitting Risk | Key Hyperparameters | Common Applications |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Linear Regression** | Supervised | Regression | Yes | Yes | Needs encoding | Recommended | No | High | High | Fast | Very Fast | Low | Low-Moderate | none core (alpha for Ridge/Lasso) | Salary/price prediction, forecasting |
| **Polynomial Regression** | Supervised | Regression | Yes | Yes | Needs encoding | Yes | Yes (via features) | High | Moderate | Fast | Very Fast | Low-Moderate | High (at high degree) | `degree` | Growth curves, diminishing returns |
| **Logistic Regression** | Supervised | Classification | Yes | Yes | Needs encoding | No | Moderate-High | High | Fast | Very Fast | Low | Low-Moderate | `C`, `penalty` | Spam, credit approval, churn |
| **KNN** | Supervised | Both | Yes | Yes | Needs encoding | Mandatory | Yes | High | Low | Instant (lazy) | Slow | High | Moderate-High (low K) | `n_neighbors`, `metric` | Recommendation, anomaly detection |
| **Decision Tree** | Supervised | Both | Yes | Yes | Yes (native) | No | Yes | Low-Moderate | High | Fast | Very Fast | Low | High (unconstrained) | `max_depth`, `min_samples_leaf` | Credit approval, medical diagnosis |
| **Random Forest** | Supervised | Both | Yes | Yes | Yes (native) | No | Yes | Low | Moderate | Moderate (parallel) | Moderate | High | Low-Moderate | `n_estimators`, `max_depth`, `max_features` | Fraud detection, risk scoring |
| **Boosting** | Supervised | Both | Yes | Yes | Yes (native, varies) | No | Yes | Moderate-High | Low | Slow (sequential) | Moderate | Moderate-High | Moderate-High (if untuned) | `n_estimators`, `learning_rate`, `max_depth` | Competitions, tabular data, credit risk |
| **K-Means** | Unsupervised | Clustering | No | Yes | Needs encoding | Mandatory | No | High | Moderate | Fast | Fast | Low | Moderate (wrong K) | `n_clusters`, `init`, `n_init` | Customer segmentation, image compression |
| **PCA** | Unsupervised | Dim. Reduction | No | Yes | Needs encoding | No (linear only) | High | Low | Moderate | Fast | Fast | Low-Moderate | `n_components` | Feature compression, visualization, noise reduction |

---

# 2. When Should I Use Which Algorithm?

## Regression Decision Flow

```
Target Variable = Continuous?
      ↓ YES
Is relationship roughly linear?
      ↓
   YES → Linear Regression
      ↓ NO
Is relationship a smooth curve?
      ↓
   YES → Polynomial Regression
      ↓ NO (complex/non-linear)
Need interpretability?
      ↓
   YES → Decision Tree (Regressor)
      ↓ NO
Need a stable, low-tuning baseline?
      ↓
   YES → Random Forest (Regressor)
      ↓ NO
Need maximum accuracy, can tune?
      ↓
   YES → Boosting (Regressor)
```

## Classification Decision Flow

```
Target Variable = Category?
      ↓ YES
Need probability output + interpretability + linear boundary OK?
      ↓
   YES → Logistic Regression
      ↓ NO
Small dataset, irregular boundary, low latency not critical?
      ↓
   YES → KNN
      ↓ NO
Need clear, explainable decision rules?
      ↓
   YES → Decision Tree
      ↓ NO
Need a robust, low-tuning baseline?
      ↓
   YES → Random Forest
      ↓ NO
Need maximum accuracy, willing to tune?
      ↓
   YES → Boosting (XGBoost/LightGBM/CatBoost)
```

## Clustering Decision Flow

```
No labels, want to discover GROUPS?
      ↓ YES
Clusters roughly spherical, similar size, K estimable?
      ↓
   YES → K-Means
      ↓ NO
Irregular shapes / unknown K / varying density?
      ↓
   YES → DBSCAN / Hierarchical Clustering (outside Phase 1 scope)
```

## Dimensionality Reduction Decision Flow

```
No labels, too many/correlated features?
      ↓ YES
Need interpretability of remaining features?
      ↓
   YES → Feature Selection (keep subset of originals)
      ↓ NO
Need maximum compression / visualization / noise reduction?
      ↓
   YES → PCA
```

---

# 3. Pairwise Comparisons

### Linear Regression vs Polynomial Regression

| Aspect | Linear Regression | Polynomial Regression |
|---|---|---|
| Relationship captured | Straight line | Curved |
| Underlying algorithm | Linear | Still Linear (on engineered features) |
| Overfitting risk | Low | High (at high degree) |
| Extrapolation | Reasonably stable | Risky, unstable |
| Scaling need | Recommended | Strongly recommended (x³ >> x) |

### Linear Regression vs Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|---|---|---|
| Task | Regression | Classification |
| Output | Continuous value | Probability (0-1) via Sigmoid |
| Cost Function | MSE | Log Loss |
| Coefficient meaning | Direct unit change in target | Change in log-odds |

### Decision Tree vs Random Forest

| Aspect | Decision Tree | Random Forest |
|---|---|---|
| Models | One | Many (ensemble) |
| Overfitting | High | Much lower |
| Stability | Low (high variance) | High |
| Interpretability | High (traceable path) | Lower (averaged) |
| Speed | Faster | Slower (but parallelizable) |

### Random Forest vs Boosting

| Aspect | Random Forest | Boosting |
|---|---|---|
| Training | Parallel, independent | Sequential, dependent |
| Primary effect | Reduces Variance | Reduces Bias |
| Overfitting from more trees | Low risk | Higher risk |
| Tuning effort | Lower | Higher |
| Typical accuracy | Strong baseline | Often higher, needs tuning |

### Decision Tree vs Boosting

| Aspect | Decision Tree | Boosting |
|---|---|---|
| Number of trees | One | Many, sequential, shallow |
| Individual tree depth | Often deep | Typically shallow ("weak learners") |
| Bias | Can be low (deep tree) | Starts high, reduced sequentially |
| Overfitting control | Stopping/pruning | learning_rate + n_estimators |

### KNN vs Logistic Regression

| Aspect | KNN | Logistic Regression |
|---|---|---|
| Learning type | Lazy (no training) | Eager (learns coefficients) |
| Decision boundary | Can be any shape | Roughly linear |
| Prediction speed | Slow | Very fast |
| Scaling | Mandatory | Recommended |
| Interpretability | Low | High |

### KNN vs Decision Tree

| Aspect | KNN | Decision Tree |
|---|---|---|
| Scaling required | Yes | No |
| Decision logic | Distance-based | Threshold/rule-based |
| Interpretability | Low | High |
| Training cost | O(1) | O(n·d·log n) |
| Prediction cost | O(n·d) | O(log n) |

### KMeans vs PCA

| Aspect | KMeans | PCA |
|---|---|---|
| Goal | Group similar points (clustering) | Compress features (dim. reduction) |
| Output | Cluster labels | New transformed features |
| Uses distance? | Yes | Uses variance, not distance directly |
| Both require scaling | Yes | Yes |

### Feature Selection vs PCA

| Aspect | Feature Selection | PCA |
|---|---|---|
| Output | Subset of original features | New blended features |
| Interpretability | High | Low |
| Uses target (y)? | Often | Never (unsupervised) |
| Information handling | Discards entire features | Compresses across all features |

### Standardization vs Normalization

| Aspect | Standardization | Normalization (Min-Max) |
|---|---|---|
| Result range | Mean 0, Std 1 (unbounded) | Fixed [0,1] |
| Outlier sensitivity | Lower | Higher |
| Default choice | Yes, generally | When bounded range specifically needed |

### Bagging vs Boosting

| Aspect | Bagging (e.g. Random Forest) | Boosting |
|---|---|---|
| Tree training | Independent, parallel | Sequential, dependent |
| Fixes | Variance | Bias |
| Base learners | Deeper trees ok | Weak, shallow learners |
| Risk of more iterations | Low | Higher (can overfit) |

### Regression vs Classification

| Aspect | Regression | Classification |
|---|---|---|
| Output | Continuous number | Discrete category |
| Metrics | MAE, MSE, RMSE, R² | Accuracy, Precision, Recall, F1, AUC |
| Example algorithms | Linear/Polynomial Regression | Logistic Regression |
| Shared algorithms | KNN, Trees, Forest, Boosting (both) | KNN, Trees, Forest, Boosting (both) |

### Supervised vs Unsupervised Learning

| Aspect | Supervised | Unsupervised |
|---|---|---|
| Labels | Required (X, y) | Not required (X only) |
| Goal | Predict known target | Discover structure |
| Evaluation | Metrics vs ground truth | Internal measures (inertia, explained variance) |
| Examples | Regression, Classification algos | KMeans, PCA |

---

# 4. Feature Scaling Guide

| Algorithm | Needs Scaling? | Why? |
|---|---|---|
| Linear Regression | Recommended | Gradient-based optimization, coefficient comparability |
| Logistic Regression | Recommended | Same as above — affects convergence and coefficient magnitude |
| KNN | **Mandatory** | Distance-based — larger-scale features dominate distance calc |
| Decision Tree | No | Threshold-based splits, unaffected by scale |
| Random Forest | No | Same reasoning — tree-based, threshold splits |
| Boosting | No | Tree-based weak learners, threshold splits |
| K-Means | **Mandatory** | Distance-based — Assignment Step relies on distance |
| PCA | **Mandatory** | Variance-based — large-scale features dominate variance calc |

---

# 5. Overfitting Guide

| Algorithm | Overfitting Risk | Underfitting Risk | Typical Fixes |
|---|---|---|---|
| Linear Regression | Low | High (if truly non-linear) | Add features, reduce regularization |
| Polynomial Regression | High (high degree) | High (low degree) | Tune degree via CV, regularize |
| Logistic Regression | Low-Moderate | High (non-linear boundary) | Add features, reduce regularization |
| KNN | High (small K) | High (large K) | Tune K via CV |
| Decision Tree | High (unconstrained) | High (very shallow) | max_depth, min_samples_leaf, pruning |
| Random Forest | Low-Moderate | Low | Tune max_depth, min_samples_leaf per tree |
| Boosting | Moderate-High (untuned) | Low (with enough rounds) | Lower learning_rate, limit n_estimators/max_depth |
| K-Means | N/A (wrong K = poor fit) | N/A | Elbow Method, Silhouette Score |
| PCA | N/A (info loss risk instead) | N/A | Check cumulative explained variance |

---

# 6. Hyperparameter Cheat Sheet

| Algorithm | Hyperparameter | Controls | Effect of Increasing |
|---|---|---|---|
| Linear Regression (Ridge/Lasso) | `alpha` (λ) | Regularization strength | Shrinks coefficients more → risk of underfitting |
| Logistic Regression | `C` | Inverse regularization strength | Higher C = less regularization = more overfitting risk |
| Polynomial Regression | `degree` | Curve flexibility | Higher = more flexible, more overfitting risk |
| KNN | `n_neighbors` (K) | Number of neighbors consulted | Higher K = smoother, more bias, less variance |
| Decision Tree | `max_depth` | Tree depth limit | Higher = more complex, more overfitting risk |
| Decision Tree | `min_samples_leaf` | Min samples per leaf | Higher = simpler, more conservative tree |
| Random Forest | `n_estimators` | Number of trees | Higher = more stable, more compute cost (not more overfitting) |
| Random Forest | `max_features` | Features considered per split | Lower = more tree diversity, weaker individual trees |
| Boosting | `n_estimators` | Number of sequential rounds | Higher = more correction, higher overfitting risk |
| Boosting | `learning_rate` | Trust given to each round's correction | Higher = faster learning, higher overfitting risk |
| Boosting | `max_depth` | Weak learner complexity | Higher = stronger learners, defeats "weak learner" premise |
| K-Means | `n_clusters` (K) | Number of clusters | Higher = tighter clusters, always lowers inertia (risk of overfitting groups) |
| K-Means | `n_init` | Number of random initialization attempts | Higher = more reliable result, more compute |
| PCA | `n_components` | Number of retained dimensions | Higher = more variance retained, less compression |

---

# 7. Complexity Cheat Sheet

| Algorithm | Training | Prediction | Memory |
|---|---|---|---|
| Linear Regression | Fast (closed-form/few GD steps) | O(d) — near-instant | Low (stores d coefficients) |
| Polynomial Regression | Fast, scales with # terms | O(d) — near-instant | Low-Moderate |
| Logistic Regression | Fast | O(d) — near-instant | Low |
| KNN | O(1) (just stores data) | O(n·d) per query — slow | High (stores entire dataset) |
| Decision Tree | O(n·d·log n) | O(log n) — very fast | Low-Moderate |
| Random Forest | N × single-tree cost (parallelizable) | N × single-tree cost | High (stores N trees) |
| Boosting | N × single-tree cost (sequential, NOT parallelizable across rounds) | N × single-tree cost | Moderate-High |
| K-Means | O(n·K·d·iterations) | O(K·d) per point | Low |
| PCA | ~O(n·d²) or O(d³) (covariance/SVD-based) | O(d·n_components) | Low-Moderate |

🎯 **Interview Tip**: KNN flips the usual tradeoff — training is instant, prediction is slow. Every other algorithm here is the opposite: invest time in training so prediction is fast.

---

# 8. Advantages vs Limitations

| Algorithm | Strengths | Weaknesses | Best Use Case | Worst Use Case |
|---|---|---|---|---|
| Linear Regression | Fast, interpretable, simple | Assumes linearity, outlier-sensitive | Simple linear relationships, baselines | Non-linear, noisy data (e.g. raw stock prices) |
| Polynomial Regression | Captures curvature, still interpretable | Overfits at high degree, poor extrapolation | Growth/diminishing-returns curves | Data needing extrapolation beyond range |
| Logistic Regression | Fast, interpretable, probabilistic output | Linear boundary only | Credit scoring, spam, quick baselines | Complex non-linear class boundaries |
| KNN | Simple, no training, flexible boundary | Slow prediction, memory heavy, curse of dimensionality | Small datasets, recommendation-style tasks | Large datasets, high-dimensional data |
| Decision Tree | Highly interpretable, no scaling, handles non-linearity | Unstable, overfits easily | Explainable rules (finance, healthcare) | Need max accuracy/stability alone |
| Random Forest | Stable, strong baseline, handles non-linearity | Less interpretable, slower than single tree | General-purpose tabular problems | Need single traceable decision path |
| Boosting | Highest accuracy potential on tabular data | Needs careful tuning, sequential (slow), overfitting risk | Competitions, high-stakes accuracy needs | Limited tuning time/resources |
| K-Means | Fast, scalable, simple | Needs K upfront, assumes spherical clusters | Customer segmentation, compression | Irregular/non-spherical clusters |
| PCA | Compresses features, fights curse of dimensionality | Less interpretable, linear only | Preprocessing for KNN/KMeans, visualization | When interpretability of features is required |

---

# 9. Business Problem → Algorithm Mapping

| Business Problem | Likely Algorithm | Why |
|---|---|---|
| **House Price Prediction** | Linear Regression → Random Forest/Boosting | Regression; start simple, escalate if relationships are non-linear |
| **Spam Detection** | Logistic Regression | Fast, interpretable, binary classification, probability useful for thresholding |
| **Customer Churn** | Random Forest / Boosting | Complex, non-linear behavioral patterns; tabular structured data |
| **Fraud Detection** | Random Forest / Boosting + Imbalanced Data techniques | Non-linear patterns, rare-event classification, needs Recall focus |
| **Movie Recommendation (High Level)** | KNN (simple version) / more advanced collaborative filtering | "Users similar to you" is a direct nearest-neighbor concept |
| **Customer Segmentation** | K-Means | Unsupervised — discover natural customer groups, no labels available |
| **Sales Forecasting** | Linear/Polynomial Regression → Boosting | Continuous target; trend/seasonality patterns often non-linear |
| **Demand Forecasting** | Regression (Linear → Boosting) | Similar to sales forecasting — continuous numeric target |
| **Feature Compression** | PCA | Unsupervised dimensionality reduction; needed before KNN/KMeans on high-dim data |

---

# 10. Interview Traps

**Q: Why not use Decision Trees everywhere?**
> A: Single trees are unstable and overfit easily. They're a great foundation, but Random Forest/Boosting (ensembles of trees) are almost always preferred in production for better accuracy and stability.

**Q: Why does KNN require scaling?**
> A: It's entirely distance-based — unscaled features with larger numeric ranges would dominate the distance calculation, making smaller-scale but potentially important features irrelevant.

**Q: Why is Random Forest more stable than a single Decision Tree?**
> A: It averages predictions across many trees trained on different bootstrap samples and feature subsets — individual trees' errors are largely uncorrelated and cancel out, reducing variance.

**Q: When is Boosting better than Random Forest?**
> A: When maximum accuracy is the priority and you have time/resources to tune hyperparameters carefully — Boosting reduces bias through sequential correction, often outperforming Random Forest on structured data.

**Q: Why is Logistic Regression still popular despite being "simple"?**
> A: It's fast, interpretable, gives calibrated probabilities, and is a strong, defensible baseline — especially valuable in regulated industries requiring explainability.

**Q: Why is PCA not Feature Selection?**
> A: PCA creates entirely new features (blended combinations of all originals) based on variance, and is unsupervised. Feature Selection keeps a subset of original, interpretable features, often guided by the target.

**Q: When would you avoid KMeans?**
> A: When clusters are non-spherical, unevenly sized/dense, or when K is genuinely unknown and hard to estimate — DBSCAN or Hierarchical Clustering are better suited there.

**Q: Why does Boosting outperform Random Forest on many benchmarks but isn't always the default choice?**
> A: It requires more careful tuning (learning_rate, n_estimators, max_depth) and trains sequentially (slower) — a poorly tuned Boosting model can underperform a well-configured Random Forest.

**Q: Why is accuracy a bad default metric?**
> A: It's misleading on imbalanced datasets, where a model can score high by simply favoring the majority class.

---

# 11. Common Beginner Mistakes

| Mistake | Why It's Wrong |
|---|---|
| Using Linear Regression for classification | Output is unbounded, not a valid probability |
| Forgetting feature scaling before KNN/KMeans/PCA | Distance/variance calculations get dominated by large-scale features |
| Using PCA when interpretability is required | Components are blended, not human-readable like original features |
| Using Random Forest/Boosting on tiny datasets | Ensemble complexity can overfit or provide no real benefit over simpler models on small data |
| Blindly using Boosting without tuning | Can overfit badly or underperform a simple Random Forest if learning_rate/n_estimators aren't tuned |
| Judging any model by training accuracy alone | Ignores overfitting — always validate on unseen/test data |
| Assuming more trees = more overfitting (Random Forest) | Only true for Boosting; Random Forest gets more stable with more trees |
| Treating KMeans cluster numbers as ordered/meaningful | Cluster labels are arbitrary — must inspect what's inside each cluster |

---

# 12. One-Page Summary

```
SUPERVISED?
  NO  → K-Means (clustering) or PCA (compression)
  YES → Continuous target? 
          YES → Regression: Linear → Polynomial → Tree → Forest → Boosting
          NO  → Classification: Logistic → KNN → Tree → Forest → Boosting
```

| Need | Pick |
|---|---|
| Fast + interpretable | Linear/Logistic Regression |
| Curved relationship | Polynomial Regression |
| Explainable rules | Decision Tree |
| Robust baseline, low tuning | Random Forest |
| Max accuracy, will tune | Boosting |
| Small data, irregular boundary | KNN |
| Discover groups (no labels) | K-Means |
| Compress features (no labels) | PCA |

**Scaling Required**: KNN, K-Means, PCA, Linear/Logistic Regression (recommended)
**Scaling NOT Required**: Decision Tree, Random Forest, Boosting

**Overfitting-Prone**: Deep single Decision Tree, high-degree Polynomial, small-K KNN, untuned Boosting
**Overfitting-Resistant**: Random Forest (more trees = more stable), regularized Linear/Logistic Regression

**Ensembles**: Random Forest (Bagging), Boosting (Sequential) — both built from Decision Trees

---

# Last Minute Interview Comparison Sheet

### ✓ Which algorithm should I use?
Use the decision flows in Section 2 — start simple (Linear/Logistic Regression), escalate to trees → ensembles only if the simple baseline underperforms.

### ✓ Which algorithms require scaling?
**Mandatory**: KNN, K-Means, PCA
**Recommended**: Linear Regression, Logistic Regression
**Not needed**: Decision Tree, Random Forest, Boosting

### ✓ Which algorithms overfit easily?
Unconstrained Decision Trees, high-degree Polynomial Regression, small-K KNN, untuned/over-trained Boosting.

### ✓ Which algorithms handle non-linearity?
KNN, Decision Tree, Random Forest, Boosting, Polynomial Regression (via engineered features). Linear/Logistic Regression do NOT (without feature engineering).

### ✓ Which algorithms are interpretable?
**High**: Linear Regression, Logistic Regression, Decision Tree
**Moderate**: Random Forest (feature importance), K-Means, PCA
**Low**: KNN, Boosting

### ✓ Which algorithms are ensembles?
Random Forest (Bagging) and Boosting (sequential). Both are built from Decision Trees as base learners.

### ✓ Which algorithms are fast?
**Fast training + fast prediction**: Linear/Logistic Regression, Decision Tree
**Fast training, slow prediction**: KNN
**Slower training (parallel), moderate prediction**: Random Forest
**Slowest training (sequential), moderate prediction**: Boosting

---

### ✓ Top 30 Comparison Facts Every ML Interview Candidate Should Know

1. Linear Regression assumes linearity; Polynomial Regression handles curves but is still "linear" in its coefficients.
2. Logistic Regression models log-odds, uses Sigmoid, trained via Log Loss (not MSE).
3. KNN is a lazy learner — no training, all computation at prediction time.
4. KNN and K-Means both require feature scaling because they're distance-based.
5. PCA requires feature scaling because it's variance-based.
6. Decision Trees, Random Forest, and Boosting do NOT require feature scaling.
7. A single Decision Tree overfits easily and is unstable (high variance).
8. Random Forest = Bagging + Random Feature Selection, applied to Decision Trees.
9. Random Forest trees train independently and in parallel.
10. Boosting trees train sequentially — cannot be parallelized across rounds.
11. Random Forest primarily reduces Variance; Boosting primarily reduces Bias.
12. More trees in Random Forest ≈ safe; more rounds in Boosting can cause overfitting.
13. Boosting's learning_rate and n_estimators are tightly linked (tradeoff).
14. Gini Impurity and Entropy both measure node "mixedness"; Gini is the faster default.
15. Feature importance in Random Forest is more reliable than in a single tree (averaged).
16. KMeans requires choosing K in advance; use Elbow Method or Silhouette Score.
17. KMeans assumes roughly spherical, similar-sized clusters.
18. K-Means++ is the standard smarter initialization (sklearn default).
19. PCA creates NEW blended features; Feature Selection keeps a SUBSET of original ones.
20. PCA is unsupervised — it never looks at the target variable.
21. Explained Variance Ratio guides how many PCA components to keep (commonly 90-95%).
22. Standardization is generally safer than Normalization when outliers are present.
23. Accuracy is misleading on imbalanced data — check Precision/Recall/F1 instead.
24. Data Leakage rule applies universally: fit any transformer on train only, apply to test.
25. Cross Validation (K=5 or K=10) gives more reliable performance estimates than a single split.
26. The Bias-Variance Tradeoff underlies every model complexity decision across all algorithms.
27. Regularization (L1/L2) directly targets variance/overfitting by penalizing large coefficients.
28. L1 (Lasso) can zero out coefficients (feature selection); L2 (Ridge) only shrinks them.
29. Every supervised algorithm in this handbook follows the same `.fit()` / `.predict()` interface.
30. Always try a simple baseline (Linear/Logistic Regression) before reaching for ensembles.