# Model Selection Guide

## The Core Question: "When Should I Use Which Algorithm?"

Before picking any algorithm, answer these questions in order.

```
START
  │
  ▼
Do I have labeled data (X → known y)?
  │
  ├── NO ──► UNSUPERVISED LEARNING
  │            │
  │            ├── Want to find groups? ──► Clustering (KMeans)
  │            └── Want to reduce features? ──► Dimensionality Reduction (PCA)
  │
  └── YES ─► SUPERVISED LEARNING
               │
               ▼
       Is the target continuous (number) or categorical (class)?
               │
               ├── CONTINUOUS ──► REGRESSION
               │        │
               │        ├── Linear relationship? ──► Linear Regression
               │        ├── Curved relationship? ──► Polynomial Regression
               │        └── Complex/non-linear? ──► Decision Tree / Random Forest / Boosting (Regressor)
               │
               └── CATEGORICAL ──► CLASSIFICATION
                        │
                        ├── Need interpretability + linear boundary? ──► Logistic Regression
                        ├── Small data, irregular boundary? ──► KNN
                        ├── Need explainable rules? ──► Decision Tree
                        ├── Need strong, stable baseline? ──► Random Forest
                        └── Need maximum accuracy, will tune? ──► Boosting (XGBoost/LightGBM/CatBoost)
```

## Regression vs Classification

| Aspect | Regression | Classification |
|---|---|---|
| Output | Continuous number | Discrete category |
| Example | House price, temperature | Spam/Not Spam, Churn/No Churn |
| Typical Metrics | MAE, MSE, RMSE, R² | Accuracy, Precision, Recall, F1, AUC |
| Algorithms Covered Here | Linear, Polynomial Regression, Tree/Forest/Boosting Regressors | Logistic Regression, KNN, Tree/Forest/Boosting Classifiers |

## Supervised vs Unsupervised

| Aspect | Supervised | Unsupervised |
|---|---|---|
| Labels | Required (X, y) | Not required (X only) |
| Goal | Predict a known target | Discover structure |
| Example Algorithms | Linear/Logistic Regression, Trees, Forests, Boosting, KNN | KMeans, PCA |
| Evaluation | Metrics vs ground truth | Internal measures (inertia, explained variance) |

---

## Linear Regression

| Attribute | Detail |
|---|---|
| **Problem Type** | Regression |
| **Works Best When** | Relationship between features and target is roughly linear; need fast, interpretable baseline |
| **Avoid When** | Relationship is clearly non-linear; heavy outlier contamination; features highly correlated without regularization |
| **Advantages** | Fast to train/predict, highly interpretable coefficients, minimal compute cost |
| **Limitations** | Assumes linearity, sensitive to outliers, assumes several statistical conditions (Ch. 5 assumptions) |
| **Requires Scaling?** | Recommended (not mandatory for correctness, but affects coefficient comparability and optimizer convergence) |
| **Handles Missing Values?** | No — must be imputed beforehand |
| **Handles Non-Linearity?** | No |
| **Sensitive to Outliers?** | Yes — high (squared error cost function) |
| **Typical Hyperparameters** | None core (Ridge/Lasso add `alpha`/λ) |
| **Time Complexity** | Training: fast, closed-form or few gradient steps. Prediction: O(d) — near-instant |

---

## Polynomial Regression

| Attribute | Detail |
|---|---|
| **Problem Type** | Regression |
| **Works Best When** | Relationship is smoothly curved (growth curves, diminishing returns, physics-based relationships) |
| **Avoid When** | Need to extrapolate beyond training data range; high risk of overfitting with limited data |
| **Advantages** | Captures curvature while staying interpretable; simple extension of Linear Regression |
| **Limitations** | High degree → severe overfitting; poor extrapolation; sensitive to outliers (same as Linear Regression) |
| **Requires Scaling?** | Yes — strongly recommended (x³ can dwarf x in magnitude) |
| **Handles Missing Values?** | No — must be imputed beforehand |
| **Handles Non-Linearity?** | Yes — via engineered polynomial terms |
| **Sensitive to Outliers?** | Yes — often worse than plain Linear Regression at high degrees |
| **Typical Hyperparameters** | `degree` |
| **Time Complexity** | Same as Linear Regression, but grows with number of polynomial terms |

---

## Logistic Regression

| Attribute | Detail |
|---|---|
| **Problem Type** | Classification (binary or multi-class via OvR/Softmax) |
| **Works Best When** | Decision boundary is roughly linear; need interpretable, probability-based output; fast baseline needed |
| **Avoid When** | Classes require a complex/non-linear boundary; strong feature interactions not manually engineered |
| **Advantages** | Interpretable coefficients, outputs calibrated probabilities, fast to train and predict |
| **Limitations** | Assumes roughly linear decision boundary, sensitive to outliers and multicollinearity |
| **Requires Scaling?** | Recommended (gradient-based optimization benefits from it) |
| **Handles Missing Values?** | No — must be imputed beforehand |
| **Handles Non-Linearity?** | No (without manual feature engineering) |
| **Sensitive to Outliers?** | Yes — moderate to high |
| **Typical Hyperparameters** | `C` (inverse regularization strength), `penalty` (L1/L2) |
| **Time Complexity** | Training: fast. Prediction: O(d) — near-instant |

---

## KNN (K-Nearest Neighbors)

| Attribute | Detail |
|---|---|
| **Problem Type** | Both Regression and Classification |
| **Works Best When** | Small-to-moderate dataset, low dimensionality, irregular/non-linear decision boundary |
| **Avoid When** | Large datasets (slow prediction), high-dimensional data (Curse of Dimensionality), real-time low-latency needs |
| **Advantages** | Simple, intuitive, no training phase, naturally handles non-linear boundaries |
| **Limitations** | Slow at prediction time, memory-intensive, degrades in high dimensions |
| **Requires Scaling?** | Yes — mandatory (distance-based) |
| **Handles Missing Values?** | No — must be imputed beforehand |
| **Handles Non-Linearity?** | Yes — naturally, via local neighborhoods |
| **Sensitive to Outliers?** | Yes — a nearby outlier directly affects local predictions |
| **Typical Hyperparameters** | `n_neighbors` (K), `metric` (Euclidean/Manhattan/Minkowski), `weights` |
| **Time Complexity** | Training: O(1). Prediction: O(n × d) per query — slow at scale |

---

## Decision Trees

| Attribute | Detail |
|---|---|
| **Problem Type** | Both Regression and Classification |
| **Works Best When** | Need interpretability/explainable rules; mixed numeric + categorical features; non-linear relationships |
| **Avoid When** | Need maximum stability/accuracy alone (prefer ensembles); very small datasets prone to unstable splits |
| **Advantages** | Highly interpretable, no scaling needed, handles non-linearity and feature interactions naturally |
| **Limitations** | Prone to overfitting unconstrained, unstable (high variance), greedy (not globally optimal) |
| **Requires Scaling?** | No |
| **Handles Missing Values?** | Varies by implementation — sklearn's default requires imputation beforehand |
| **Handles Non-Linearity?** | Yes — naturally |
| **Sensitive to Outliers?** | Low to moderate — outliers can still affect specific splits, but less than distance/gradient-based models |
| **Typical Hyperparameters** | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` (gini/entropy) |
| **Time Complexity** | Training: O(n × d × log n). Prediction: O(log n) — very fast |

---

## Random Forest

| Attribute | Detail |
|---|---|
| **Problem Type** | Both Regression and Classification |
| **Works Best When** | Need a strong, robust baseline with minimal tuning; non-linear data; want reduced variance vs a single tree |
| **Avoid When** | Need maximum interpretability (single clean decision path); extremely tight latency/memory budgets |
| **Advantages** | Reduces overfitting vs single tree, handles non-linearity well, provides reliable feature importance, minimal tuning needed |
| **Limitations** | Less interpretable than a single tree, slower to train/predict than a single tree, larger memory footprint |
| **Requires Scaling?** | No |
| **Handles Missing Values?** | No (standard sklearn implementation) — must be imputed beforehand |
| **Handles Non-Linearity?** | Yes — naturally |
| **Sensitive to Outliers?** | Low — averaging across many trees dampens individual outlier influence |
| **Typical Hyperparameters** | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| **Time Complexity** | Training: N × single-tree cost (parallelizable). Prediction: N × single-tree cost |

---

## Boosting (Gradient Boosting / XGBoost / LightGBM / CatBoost)

| Attribute | Detail |
|---|---|
| **Problem Type** | Both Regression and Classification |
| **Works Best When** | Maximum predictive accuracy needed on structured/tabular data; willing to invest in careful tuning |
| **Avoid When** | Limited time/resources for tuning; need fast, parallel training; need strong interpretability |
| **Advantages** | Frequently top-performing on tabular data, reduces bias via sequential correction, flexible and powerful |
| **Limitations** | Prone to overfitting if over-tuned/over-trained, cannot parallelize across the tree sequence, more hyperparameter-sensitive |
| **Requires Scaling?** | No |
| **Handles Missing Values?** | Often yes natively (XGBoost, LightGBM, CatBoost) — varies by implementation |
| **Handles Non-Linearity?** | Yes — naturally (tree-based weak learners) |
| **Sensitive to Outliers?** | Moderate to high — sequential error correction can over-focus on noisy/mislabeled points |
| **Typical Hyperparameters** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| **Time Complexity** | Training: sequential, N rounds × tree cost (not parallelizable across rounds). Prediction: N × single-tree cost |

---

## KMeans

| Attribute | Detail |
|---|---|
| **Problem Type** | Unsupervised — Clustering |
| **Works Best When** | Need fast, scalable clustering; clusters are roughly spherical and similarly sized; K is known or can be estimated |
| **Avoid When** | Clusters are irregular/non-spherical shapes; clusters have very different sizes/densities; K is unknown and hard to estimate |
| **Advantages** | Fast, simple, scalable to large datasets, easy to interpret cluster assignments |
| **Limitations** | Must choose K in advance, sensitive to initialization and outliers, assumes spherical clusters |
| **Requires Scaling?** | Yes — mandatory (distance-based) |
| **Handles Missing Values?** | No — must be imputed beforehand |
| **Handles Non-Linearity?** | No — assumes roughly convex/spherical cluster shapes |
| **Sensitive to Outliers?** | Yes — centroids are simple averages, easily pulled by extreme points |
| **Typical Hyperparameters** | `n_clusters` (K), `init` (k-means++), `n_init` |
| **Time Complexity** | O(n × K × d × iterations) — scales roughly linearly, efficient for large n |

---

## PCA

| Attribute | Detail |
|---|---|
| **Problem Type** | Unsupervised — Dimensionality Reduction |
| **Works Best When** | Many correlated features; need to fight Curse of Dimensionality before KNN/KMeans; need visualization or noise reduction |
| **Avoid When** | Interpretability of original features is required; relationships are purely non-linear; features are already largely uncorrelated |
| **Advantages** | Reduces dimensionality while preserving variance, speeds up downstream training, aids visualization and noise reduction |
| **Limitations** | Reduces interpretability (blended components), assumes linear structure, can discard task-relevant low-variance information |
| **Requires Scaling?** | Yes — mandatory (variance-based) |
| **Handles Missing Values?** | No — must be imputed beforehand |
| **Handles Non-Linearity?** | No — finds only linear combinations of features |
| **Sensitive to Outliers?** | Yes — outliers can distort variance/direction calculations |
| **Typical Hyperparameters** | `n_components` |
| **Time Complexity** | Roughly O(n × d²) or O(d³) depending on method (covariance/SVD-based) — mainly depends on feature count |

---

# Algorithm Selection Cheat Table

| Algorithm | Type | Scaling? | Missing Values? | Non-Linear? | Outlier Sensitive? | Interpretability | Training Speed | Prediction Speed |
|---|---|---|---|---|---|---|---|---|
| **Linear Regression** | Regression | Recommended | No | No | High | High | Fast | Very Fast |
| **Polynomial Regression** | Regression | Yes | No | Yes | High | Moderate | Fast | Very Fast |
| **Logistic Regression** | Classification | Recommended | No | No | Moderate-High | High | Fast | Very Fast |
| **KNN** | Both | Yes (mandatory) | No | Yes | High | Low | Instant (lazy) | Slow |
| **Decision Tree** | Both | No | Varies | Yes | Low-Moderate | High | Fast | Very Fast |
| **Random Forest** | Both | No | No (sklearn) | Yes | Low | Moderate | Moderate (parallel) | Moderate |
| **Boosting (XGBoost etc.)** | Both | No | Often Yes | Yes | Moderate-High | Low | Slow (sequential) | Moderate |
| **KMeans** | Unsupervised (Clustering) | Yes (mandatory) | No | No | High | Moderate | Fast | Fast |
| **PCA** | Unsupervised (Dim. Reduction) | Yes (mandatory) | No | No | High | Low | Moderate | Fast |

## Quick Decision Rules

- **Need interpretability above all else?** → Linear/Logistic Regression or a single Decision Tree
- **Need maximum accuracy on tabular data, can tune?** → Boosting (XGBoost/LightGBM/CatBoost)
- **Need a strong baseline with minimal tuning?** → Random Forest
- **Small dataset, irregular decision boundary?** → KNN
- **Curved but simple relationship?** → Polynomial Regression
- **No labels, want groups?** → KMeans
- **No labels, too many correlated features?** → PCA
- **High-dimensional data before KNN/KMeans?** → Apply PCA first
- **Imbalanced classification target?** → Any classifier above + techniques from Imbalanced Data chapter (class weights, resampling, threshold tuning)