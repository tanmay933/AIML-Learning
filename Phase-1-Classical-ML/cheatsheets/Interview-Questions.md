# Machine Learning Interview Questions

---

## Chapter 1: ML Fundamentals

### Beginner Questions

**Q: What is the difference between AI, ML, and Deep Learning?**
**Answer**: AI is any system that mimics intelligent behavior. ML is a subset of AI that learns patterns from data instead of explicit rules. Deep Learning is a subset of ML using multi-layer neural networks.
**Common Mistake**: Treating AI and ML as synonyms.
**Interview Tip**: Give an example of AI that isn't ML (e.g., a rule-based chess engine or pathfinding algorithm).

**Q: What is the difference between Supervised and Unsupervised Learning?**
**Answer**: Supervised learning uses labeled data (X, y) to learn a mapping to known outputs. Unsupervised learning uses unlabeled data (X only) to discover structure, like clusters.
**Common Mistake**: Assuming unsupervised learning has no practical use.
**Interview Tip**: Mention clustering (KMeans) and dimensionality reduction (PCA) as examples.

**Q: What's the difference between Regression and Classification?**
**Answer**: Regression predicts a continuous number; classification predicts a discrete category.
**Common Mistake**: Confusing "predicting a probability" with regression — it's still classification.
**Interview Tip**: Give one example each: house price (regression), spam detection (classification).

### Intermediate Questions

**Q: What is the difference between a Parameter and a Hyperparameter?**
**Answer**: Parameters are learned automatically during training (e.g., coefficients). Hyperparameters are set manually before training (e.g., max tree depth).
**Common Mistake**: Confusing the two, especially with regularization strength (a hyperparameter).
**Interview Tip**: Use the test: "if it's learned from data, it's a parameter."

**Q: What is the difference between a Model and an Algorithm?**
**Answer**: An algorithm is the general learning procedure; a model is the specific trained result (algorithm + learned parameters).
**Common Mistake**: Using the terms interchangeably in an interview.
**Interview Tip**: Analogy — algorithm is the recipe, model is the baked cake.

**Q: Why do we need a test set? Why not train on all the data?**
**Answer**: A test set simulates unseen, real-world data, letting us estimate generalization rather than memorization.
**Common Mistake**: Believing high training accuracy alone proves a good model.
**Interview Tip**: Mention data leakage as the risk if the test set is contaminated.

### Frequently Asked Questions

**Q: What is generalization?**
**Answer**: A model's ability to perform well on new, unseen data, not just training data.
**Common Mistake**: Equating generalization with training performance.
**Interview Tip**: Tie this directly to overfitting.

**Q: Explain the standard ML workflow.**
**Answer**: Problem Definition → Collect Data → Understand → Prepare → Split → Choose Model → Train → Validate → Final Evaluation → Deploy → Monitor.
**Common Mistake**: Treating this as strictly linear rather than iterative.
**Interview Tip**: Emphasize that "Monitor" is often forgotten but critical in production.

### Follow-up Questions

**Q: If a model has 99% training accuracy, is it a good model?**
**Answer**: Not necessarily — it might be overfitting. Must check test performance.
**Interview Tip**: Bridge into Bias-Variance (Chapter 7).

**Q: Is training vs inference the same as fitting vs predicting?**
**Answer**: Yes — training/fitting learns parameters; inference/predicting applies them to new data.

---

## Chapter 2: Data Preparation

### Beginner Questions

**Q: Why is data preparation important?**
**Answer**: Model quality is fundamentally limited by data quality — "garbage in, garbage out."
**Interview Tip**: Mention that most real project time goes into data prep, not modeling.

**Q: What is the difference between Mean and Median Imputation?**
**Answer**: Mean imputation is sensitive to outliers; median is robust to skew and outliers.
**Common Mistake**: Always defaulting to mean regardless of distribution shape.

**Q: What is Label Encoding vs One-Hot Encoding?**
**Answer**: Label Encoding assigns integers (good for ordinal data); One-Hot Encoding creates binary columns (good for nominal data).
**Common Mistake**: Label-encoding nominal categories, implying a false order.

### Intermediate Questions

**Q: What is Data Leakage, and how do you prevent it?**
**Answer**: Leakage is when information unavailable at prediction time influences training, inflating performance. Prevented by fitting all transformers only on training data, after the split.
**Interview Tip**: Give the scaling-before-split example as the classic case.

**Q: Standardization vs Normalization — what's the difference?**
**Answer**: Standardization centers to mean 0/std 1, robust to outliers. Normalization rescales to [0,1], sensitive to outliers.

**Q: Do Decision Trees require feature scaling?**
**Answer**: No — trees split via threshold comparisons, unaffected by scale.

### Frequently Asked Questions

**Q: What are outliers, and should they always be removed?**
**Answer**: Data points far from the rest of the distribution. Should NOT always be removed — depends on whether they're errors or genuine rare events (domain knowledge required).

**Q: What is Feature Engineering, and why does it matter?**
**Answer**: Creating/transforming features to expose more meaningful patterns. Often improves performance more than switching algorithms.

### Follow-up Questions

**Q: Why must duplicates be removed before training?**
**Answer**: Duplicates bias the model by over-representing certain data points, and can also cause train/test leakage if they span the split.

**Q: What's the correct order: split first or scale first?**
**Answer**: Always split first, then fit scalers/encoders on training data only.

---

## Chapter 3: Linear Regression

### Beginner Questions

**Q: What is Linear Regression?**
**Answer**: A supervised algorithm that fits the best straight line (or hyperplane) to predict a continuous target from features.

**Q: What do the coefficient and intercept represent?**
**Answer**: The coefficient is the change in target per unit change in a feature; the intercept is the baseline prediction when all features are 0.

### Intermediate Questions

**Q: How does Linear Regression "learn"?**
**Answer**: An optimization algorithm (typically Gradient Descent) adjusts coefficients to minimize a cost function, usually MSE.

**Q: Why does Linear Regression use MSE as a cost function?**
**Answer**: Squaring residuals prevents cancellation of positive/negative errors and penalizes larger errors more heavily.

**Q: What's the difference between Simple and Multiple Linear Regression?**
**Answer**: Simple uses one feature (a line); Multiple uses several features (a hyperplane), with one coefficient per feature.

### Frequently Asked Questions

**Q: What's a key weakness of Linear Regression?**
**Answer**: Assumes linearity, and is sensitive to outliers due to the squared-error cost function.

**Q: Would you use Linear Regression to predict stock prices?**
**Answer**: Generally no — stock prices are highly non-linear, noisy, and non-stationary, violating core assumptions.
**Interview Tip**: This is a common "gotcha" question testing if you overuse Linear Regression.

### Follow-up Questions

**Q: Does Linear Regression require feature scaling?**
**Answer**: Not strictly mandatory, but recommended for gradient-based optimization and coefficient comparability.

**Q: How do you interpret a coefficient when features are correlated?**
**Answer**: Interpretation becomes less reliable — this is the multicollinearity issue, covered fully in Chapter 5.

---

## Chapter 4: Regression Metrics

### Beginner Questions

**Q: What is a residual?**
**Answer**: The difference between actual and predicted value (actual − predicted).

**Q: What's the difference between MAE and MSE?**
**Answer**: MAE averages absolute residuals (robust); MSE averages squared residuals (penalizes large errors, sensitive to outliers).

### Intermediate Questions

**Q: Why do we use RMSE instead of just MSE?**
**Answer**: RMSE takes the square root, returning to the target's original units, making it more interpretable while retaining MSE's outlier sensitivity.

**Q: What does an R² of 0.9 mean?**
**Answer**: The model explains 90% of the variance in the target compared to a baseline that always predicts the mean.

**Q: Can R² be negative?**
**Answer**: Yes — it means the model performs worse than simply predicting the mean every time.

### Frequently Asked Questions

**Q: Why does Adjusted R² exist?**
**Answer**: Plain R² always increases with more features, even irrelevant ones. Adjusted R² penalizes unnecessary features.

**Q: If RMSE is much higher than MAE, what does that suggest?**
**Answer**: A few large errors (outliers) are disproportionately inflating RMSE.

### Follow-up Questions

**Q: Which metric is most robust to outliers?**
**Answer**: MAE, since it doesn't square residuals.

**Q: How would you choose between MAE and RMSE in a real project?**
**Answer**: Depends on whether large errors are especially costly to the business — if so, use RMSE; otherwise MAE.

---

## Chapter 5: Linear Regression Assumptions

### Beginner Questions

**Q: Why does Linear Regression rely on assumptions?**
**Answer**: They ensure coefficients and statistical guarantees (confidence intervals, significance) are reliable, not just the raw predictions.

**Q: What is multicollinearity?**
**Answer**: When two or more features are highly correlated, making individual coefficients unstable and hard to interpret.

### Intermediate Questions

**Q: How do you check if the linearity assumption holds?**
**Answer**: Plot residuals vs predicted values — a random scatter suggests linearity; a curved pattern suggests it doesn't.

**Q: What is heteroscedasticity, and how is it detected?**
**Answer**: Non-constant variance of residuals across predicted values; detected via a funnel-shaped residual plot.

**Q: What's the difference between correlation and causation?**
**Answer**: Correlation means variables move together statistically; causation means one directly produces a change in the other. Regression only shows correlation.

### Frequently Asked Questions

**Q: Does violating an assumption always make the model unusable?**
**Answer**: No — some violations (mild non-normality) mainly affect statistical inference, not predictive accuracy. Others (non-linearity) hurt predictions directly.

**Q: How is independence of observations typically violated?**
**Answer**: Common in time-series data, where residuals are correlated across time (autocorrelation).

### Follow-up Questions

**Q: If your residual plot shows a funnel shape, what does that indicate?**
**Answer**: Heteroscedasticity — inconsistent error variance across prediction ranges.

**Q: Does a high R² guarantee assumptions are satisfied?**
**Answer**: No — aggregate metrics can look fine even when assumptions are clearly violated.

---

## Chapter 6: Polynomial Regression

### Beginner Questions

**Q: Is Polynomial Regression a different algorithm from Linear Regression?**
**Answer**: No — it's still Linear Regression, applied to engineered polynomial features (x, x², x³, ...).

**Q: What happens if you choose too high a polynomial degree?**
**Answer**: The model overfits — very low training error but poor test performance.

### Intermediate Questions

**Q: What happens if you choose too low a degree for curved data?**
**Answer**: The model underfits — high error on both training and test data.

**Q: How do you choose the right polynomial degree in practice?**
**Answer**: Compare training vs test performance across degrees, and pick the degree with the best test performance, ideally validated with Cross Validation.

**Q: Why is Polynomial Regression risky for extrapolation?**
**Answer**: Polynomial curves can behave unpredictably outside the training data's range.

### Frequently Asked Questions

**Q: Does Polynomial Regression require feature scaling?**
**Answer**: Yes, more so than plain Linear Regression, since higher-degree terms can have vastly larger magnitudes.

### Follow-up Questions

**Q: Why might a degree-15 polynomial have lower training error but be a worse model than degree-2?**
**Answer**: It's overfitting — fitting noise instead of the true underlying pattern.

---

## Chapter 7: Bias vs Variance

### Beginner Questions

**Q: What is the difference between bias and variance?**
**Answer**: Bias is error from overly simple assumptions (underfitting); variance is error from being overly sensitive to training data (overfitting).

**Q: What is the Bias-Variance Tradeoff?**
**Answer**: Reducing bias (more complexity) typically increases variance, and vice versa — the goal is to balance both, not eliminate either.

### Intermediate Questions

**Q: How do you diagnose overfitting vs underfitting?**
**Answer**: Compare training and test error. Similar high error = underfitting. Low training error, high test error = overfitting.

**Q: Does adding more training data always help?**
**Answer**: It mainly helps reduce variance (overfitting); it does little for high bias, since a too-simple model stays too simple regardless of data volume.

**Q: If a model has 99% training accuracy and 70% test accuracy, what's happening?**
**Answer**: Classic overfitting (high variance).

### Frequently Asked Questions

**Q: Name two ways to reduce variance.**
**Answer**: Simplify the model, gather more data; also Regularization, Cross Validation, and Ensembles.

**Q: Is the bias-variance tradeoff specific to regression?**
**Answer**: No — it's universal, applying to classification, clustering, and neural networks.

### Follow-up Questions

**Q: What's the relationship between model complexity and bias/variance?**
**Answer**: Simpler models → higher bias, lower variance. Complex models → lower bias, higher variance.

---

## Chapter 8: Regularization

### Beginner Questions

**Q: Why does Regularization exist?**
**Answer**: To reduce overfitting (variance) by discouraging excessively large weights, constraining model complexity.

**Q: What's the difference between L1 and L2 Regularization?**
**Answer**: L1 (Lasso) can zero out coefficients (feature selection); L2 (Ridge) shrinks coefficients smoothly without eliminating them.

### Intermediate Questions

**Q: When would you choose Lasso over Ridge?**
**Answer**: When you suspect many features are irrelevant and want automatic feature selection.

**Q: When would you choose Ridge over Lasso?**
**Answer**: When most features are relevant, especially with multicollinearity — Ridge stabilizes coefficients without discarding correlated features.

**Q: What is Elastic Net?**
**Answer**: A combination of L1 and L2 penalties, balancing feature selection with coefficient stability.

### Frequently Asked Questions

**Q: What happens if regularization strength (λ) is too high?**
**Answer**: The penalty dominates the cost function, shrinking coefficients excessively, causing underfitting.

**Q: Why is feature scaling important before Regularization?**
**Answer**: The penalty is based on coefficient magnitude — unscaled features would be penalized unfairly relative to their natural scale.

### Follow-up Questions

**Q: Does Regularization change the optimization algorithm?**
**Answer**: No — it changes the cost function being minimized (adds a penalty term); the training process itself works the same way.

---

## Chapter 9: Cross Validation

### Beginner Questions

**Q: Why is a single train/test split sometimes not enough?**
**Answer**: Performance depends heavily on which rows land in the test set — a different split can give a noticeably different result.

**Q: How does K-Fold Cross Validation work?**
**Answer**: Split data into K folds; train K times, each using a different fold as test and the rest as training; average the results.

### Intermediate Questions

**Q: What's the difference between K-Fold and Stratified K-Fold?**
**Answer**: Stratified K-Fold preserves class proportions in each fold — important for imbalanced classification.

**Q: What is LOOCV?**
**Answer**: Leave-One-Out Cross Validation — K-Fold with K = number of samples; low bias, high compute cost, mainly for small datasets.

**Q: Why would you use Nested Cross Validation?**
**Answer**: To avoid leakage when both tuning hyperparameters and estimating final performance — an inner loop tunes, an outer loop gives an unbiased estimate.

### Frequently Asked Questions

**Q: How do you choose K?**
**Answer**: K=5 or K=10 are the standard practical defaults, balancing reliability and compute cost.

**Q: Does Cross Validation eliminate the need for a separate test set?**
**Answer**: No — CV is for tuning/comparison; a final held-out test set is still needed for an unbiased final check.

### Follow-up Questions

**Q: What's a common way engineers accidentally leak data during CV?**
**Answer**: Fitting preprocessing (scalers/encoders) on the entire dataset before running Cross Validation, instead of per-fold.

---

## Chapter 10: Logistic Regression

### Beginner Questions

**Q: Why can't you use Linear Regression for classification?**
**Answer**: Its output is unbounded and doesn't represent a valid probability (can be <0 or >1).

**Q: What does the Sigmoid function do?**
**Answer**: Transforms any real number into a probability between 0 and 1.

### Intermediate Questions

**Q: What is Logistic Regression actually modeling?**
**Answer**: The log-odds of the positive class, as a linear function of features; Sigmoid converts that back to a probability.

**Q: How do you interpret a coefficient in Logistic Regression?**
**Answer**: A positive coefficient increases the log-odds (and probability) of the positive class; it doesn't directly translate to a fixed change in outcome like in Linear Regression.

**Q: What cost function does Logistic Regression use?**
**Answer**: Log Loss (Binary Cross-Entropy), not MSE — it's designed for probability outputs.

### Frequently Asked Questions

**Q: Is the 0.5 threshold always correct?**
**Answer**: No — it's a default; the threshold is a business decision, often adjusted (e.g., lowered for fraud/disease detection to increase recall).

**Q: How does Logistic Regression handle multi-class problems?**
**Answer**: One-vs-Rest (multiple binary classifiers) or Softmax (multinomial) Regression.

### Follow-up Questions

**Q: What's the difference between `.predict()` and `.predict_proba()`?**
**Answer**: `.predict()` applies the default threshold to return a class label; `.predict_proba()` returns the underlying probability.

---

## Chapter 11: Classification Metrics

### Beginner Questions

**Q: Why is accuracy sometimes misleading?**
**Answer**: On imbalanced data, a model can achieve high accuracy just by predicting the majority class, missing the minority class entirely.

**Q: What's the difference between Precision and Recall?**
**Answer**: Precision measures how many flagged positives are correct; Recall measures how many actual positives were caught.

### Intermediate Questions

**Q: When would you prioritize Recall over Precision?**
**Answer**: When false negatives are costlier — e.g., disease screening or fraud detection.

**Q: What is F1 Score, and why the harmonic mean?**
**Answer**: F1 balances Precision and Recall; harmonic mean heavily penalizes cases where one is very low, unlike a simple average.

**Q: What does AUC represent?**
**Answer**: The probability the model ranks a random positive higher than a random negative — overall ranking ability across all thresholds.

### Frequently Asked Questions

**Q: When would you prefer Precision-Recall Curve over ROC?**
**Answer**: On heavily imbalanced datasets, since ROC/AUC can look overly optimistic due to a large number of true negatives.

**Q: If a fraud model has 99% accuracy, is it good?**
**Answer**: Not necessarily — if fraud is rare, always predicting "not fraud" also hits 99% accuracy while catching zero fraud.

### Follow-up Questions

**Q: What's the relationship between Specificity and FPR?**
**Answer**: FPR = 1 − Specificity.

---

## Chapter 12: Imbalanced Data

### Beginner Questions

**Q: What is imbalanced data?**
**Answer**: When one class vastly outnumbers another — common because rare, high-value events (fraud, disease) are rare by nature.

**Q: Why does accuracy fail on imbalanced data?**
**Answer**: A model can hit high accuracy by always predicting the majority class, missing the minority class entirely.

### Intermediate Questions

**Q: What's the difference between under-sampling and over-sampling?**
**Answer**: Under-sampling removes majority examples (loses data); over-sampling duplicates minority examples (risks overfitting).

**Q: What is SMOTE?**
**Answer**: Synthetic Minority Over-sampling Technique — generates new synthetic minority examples by interpolation, reducing overfitting vs naive duplication.

**Q: How do class weights help with imbalance?**
**Answer**: They modify the cost function to penalize minority-class mistakes more heavily, without touching the dataset itself.

### Frequently Asked Questions

**Q: Why should resampling only be applied to the training set?**
**Answer**: The test set must reflect the real-world class distribution; resampling it (or before splitting) leaks information and invalidates evaluation.

**Q: What's a lower-cost alternative to resampling?**
**Answer**: Adjusting the decision threshold — no retraining or data changes required.

### Follow-up Questions

**Q: Which metrics should you avoid/use for imbalanced data?**
**Answer**: Avoid Accuracy; use Precision, Recall, F1, and Precision-Recall curves.

---

## Chapter 13: KNN

### Beginner Questions

**Q: Why is KNN called a lazy learner?**
**Answer**: It doesn't build a model during training — it stores the data and defers computation to prediction time.

**Q: How does KNN make a prediction?**
**Answer**: Finds the K nearest training points and takes a majority vote (classification) or average (regression).

### Intermediate Questions

**Q: Why does KNN require feature scaling?**
**Answer**: It relies entirely on distance — unscaled features with larger ranges would dominate the distance calculation.

**Q: How does K relate to bias-variance?**
**Answer**: Small K → high variance (overfitting); large K → high bias (underfitting).

**Q: What is the Curse of Dimensionality?**
**Answer**: As feature count grows, distances between points converge, making "nearest" neighbors less meaningful.

### Frequently Asked Questions

**Q: What's KNN's time complexity at prediction?**
**Answer**: Roughly O(n×d) per prediction — must compute distance to every training point.

**Q: How is KNN used for regression?**
**Answer**: Instead of voting, it averages the target values of the K nearest neighbors.

### Follow-up Questions

**Q: What's a major limitation of KNN vs Logistic Regression?**
**Answer**: KNN is much slower and more memory-intensive at prediction, since it stores and searches the entire dataset.

---

## Chapter 14: Decision Trees

### Beginner Questions

**Q: How does a Decision Tree decide what to split on?**
**Answer**: It evaluates all possible splits and picks the one with the highest Information Gain (greatest reduction in Gini/Entropy).

**Q: What's the difference between Gini Impurity and Entropy?**
**Answer**: Both measure class "mixedness"; Gini is computationally simpler (no logarithms) and is the common default.

### Intermediate Questions

**Q: Why don't Decision Trees require feature scaling?**
**Answer**: Splits use threshold comparisons on individual features, unaffected by scale.

**Q: How do you prevent a Decision Tree from overfitting?**
**Answer**: Use stopping criteria (max_depth, min_samples_leaf) or prune the tree after growing it fully.

**Q: What is feature importance in a Decision Tree?**
**Answer**: A measure of how much each feature contributed to reducing impurity across all its splits.

### Frequently Asked Questions

**Q: Why is a single Decision Tree considered "unstable"?**
**Answer**: Small changes in training data can produce a substantially different tree structure — high variance.

**Q: How does a Regression Tree differ from a Classification Tree?**
**Answer**: It splits to minimize variance (not maximize purity) and predicts the average target value at each leaf.

### Follow-up Questions

**Q: Why are Decision Trees a good foundation for Random Forest?**
**Answer**: A single tree's instability/overfitting can be addressed by combining many trees and averaging — exactly what Random Forest does.

---

## Chapter 15: Random Forest

### Beginner Questions

**Q: What problem does Random Forest solve?**
**Answer**: It addresses a single Decision Tree's instability and overfitting by combining many trees trained on different data/feature subsets.

**Q: What are the two sources of randomness in Random Forest?**
**Answer**: Bootstrap sampling of rows and random feature selection at each split.

### Intermediate Questions

**Q: What is Out-of-Bag (OOB) error?**
**Answer**: A free validation estimate using the ~37% of data left out of each tree's bootstrap sample.

**Q: How does Random Forest reduce variance?**
**Answer**: Individual trees still overfit, but their errors are largely uncorrelated — averaging cancels out much of the noise.

**Q: Does increasing n_estimators cause overfitting?**
**Answer**: No — more trees generally stabilize the ensemble; the main cost is compute time, not overfitting.

### Frequently Asked Questions

**Q: Why can Random Forest training be parallelized while Boosting cannot?**
**Answer**: Each Random Forest tree is trained independently; Boosting trees depend sequentially on previous ones.

**Q: Is Random Forest more or less interpretable than a single Decision Tree?**
**Answer**: Less — no single traceable decision path, though feature importance remains available (averaged across trees).

### Follow-up Questions

**Q: Does Random Forest require feature scaling?**
**Answer**: No — same threshold-based splitting logic as individual trees.

---

## Chapter 16: Boosting

### Beginner Questions

**Q: What is the core idea behind Boosting?**
**Answer**: Sequentially train weak learners, each correcting the errors of the previous ones, then combine them into one strong model.

**Q: What is a weak learner?**
**Answer**: A model only slightly better than random guessing — used deliberately as Boosting's building block.

### Intermediate Questions

**Q: What's the difference between AdaBoost and Gradient Boosting?**
**Answer**: AdaBoost reweights misclassified data points; Gradient Boosting fits new learners directly to residual errors.

**Q: How does Boosting differ from Random Forest structurally?**
**Answer**: Random Forest trains trees independently/in parallel; Boosting trains sequentially, each depending on prior errors.

**Q: From a bias-variance perspective, how do they differ?**
**Answer**: Random Forest primarily reduces variance (averaging); Boosting primarily reduces bias (sequential correction).

### Frequently Asked Questions

**Q: Can Boosting overfit?**
**Answer**: Yes — unlike Random Forest, too many rounds or too high a learning rate can cause overfitting.

**Q: What's the relationship between learning_rate and n_estimators?**
**Answer**: Smaller learning rate needs more rounds but often generalizes better; higher learning rate learns faster but risks overfitting.

### Follow-up Questions

**Q: When would you choose XGBoost over Random Forest?**
**Answer**: When maximum accuracy is the priority and you're willing to invest in careful hyperparameter tuning.

---

## Chapter 17: KMeans

### Beginner Questions

**Q: How does K-Means work, step by step?**
**Answer**: Initialize K centroids, repeat: assign points to nearest centroid, recalculate centroids as the average of assigned points, until convergence.

**Q: How do you choose K?**
**Answer**: Elbow Method (inertia vs K) or Silhouette Score (accounts for cluster separation too).

### Intermediate Questions

**Q: Why is K-Means sensitive to initialization?**
**Answer**: It only guarantees a local optimum — a poor starting position can lead to a suboptimal clustering.

**Q: What is K-Means++?**
**Answer**: A smarter initialization strategy that spreads out starting centroids, reducing the risk of poor convergence.

**Q: Why does K-Means require feature scaling?**
**Answer**: It's distance-based — larger-scale features would dominate the Assignment Step's distance calculations.

### Frequently Asked Questions

**Q: What are the main limitations of K-Means?**
**Answer**: Must choose K in advance, assumes spherical/similar-sized clusters, sensitive to outliers and initialization.

**Q: How is K-Means different from supervised algorithms?**
**Answer**: It has no labels — only uses X to discover structure.

### Follow-up Questions

**Q: What would you use instead of K-Means for non-spherical clusters?**
**Answer**: DBSCAN or Hierarchical Clustering.

---

## Chapter 18: PCA

### Beginner Questions

**Q: Why does PCA exist?**
**Answer**: To reduce feature count while preserving as much meaningful information (variance) as possible, fighting the Curse of Dimensionality.

**Q: What is a principal component?**
**Answer**: A new feature — a weighted combination of original features, chosen to capture as much variance as possible.

### Intermediate Questions

**Q: Why does PCA require feature scaling?**
**Answer**: PCA looks for directions of maximum variance; larger-scale features would dominate that calculation regardless of actual informativeness.

**Q: What does "explained variance" mean?**
**Answer**: The percentage of total dataset variance a given component captures; used to decide how many components to keep.

**Q: What's the difference between PCA and Feature Selection?**
**Answer**: Feature Selection keeps a subset of original features (often using the target); PCA creates entirely new blended features, unsupervised.

### Frequently Asked Questions

**Q: Is PCA supervised or unsupervised?**
**Answer**: Unsupervised — it never considers the target variable.

**Q: What's a key limitation of PCA?**
**Answer**: Reduced interpretability (blended components), assumes linear relationships, can discard task-relevant information.

### Follow-up Questions

**Q: How would you decide how many components to keep?**
**Answer**: Cumulative explained variance threshold (e.g., 90-95%) or a scree plot's "elbow" point.

---

## Chapter 19: End-to-End ML Project

### Beginner Questions

**Q: What's the first step in any ML project?**
**Answer**: Problem Definition — clarifying the business problem, whether ML is needed, and the success metric.

**Q: Why must the evaluation metric be chosen before modeling?**
**Answer**: The metric reflects the real business cost of different error types, which should drive every downstream decision.

### Intermediate Questions

**Q: Walk me through your approach to an ML project.**
**Answer**: Define problem/metric → explore & clean data → engineer features avoiding leakage → split → baseline model → Cross Validate/tune → select model by the right metric → error analysis → deploy → monitor.

**Q: Why is the ML pipeline iterative rather than linear?**
**Answer**: Error analysis often reveals gaps that send you back to feature engineering or even problem definition.

### Frequently Asked Questions

**Q: What is model drift?**
**Answer**: When real-world data patterns shift over time, degrading a deployed model's performance — detected via monitoring.

**Q: Why is error analysis important beyond aggregate metrics?**
**Answer**: It reveals specific patterns in what the model gets wrong, often pointing to a feature engineering gap rather than a modeling gap.

### Follow-up Questions

**Q: How does this Phase 1 foundation connect to Deep Learning?**
**Answer**: The same core ideas — cost functions, optimization, train/test discipline, bias-variance — still apply; Deep Learning adds new tools on top of this foundation, not a replacement for it.

---

# Top 50 Most Important Interview Questions

1. What is the difference between AI, ML, and Deep Learning?
2. What is the difference between Supervised and Unsupervised Learning?
3. What is the difference between Regression and Classification?
4. What's the difference between a Parameter and a Hyperparameter?
5. Why do we need a test/validation set?
6. What is generalization, and why does it matter?
7. Why is data preparation more important than algorithm choice?
8. What is Data Leakage, and how do you prevent it?
9. Mean vs Median Imputation — when to use each?
10. One-Hot Encoding vs Label Encoding — when to use each?
11. Standardization vs Normalization — what's the difference?
12. Do Decision Trees require feature scaling? Why or why not?
13. What is Linear Regression's cost function, and why MSE?
14. What's a key weakness of Linear Regression?
15. What's the difference between MAE, MSE, and RMSE?
16. Can R² be negative? What does that mean?
17. Why does Adjusted R² exist?
18. What are the assumptions behind Linear Regression?
19. What is heteroscedasticity, and how do you detect it?
20. What's the difference between correlation and causation?
21. Why is Polynomial Regression "still Linear Regression"?
22. How do you choose the right polynomial degree?
23. What is the Bias-Variance Tradeoff?
24. How do you diagnose overfitting vs underfitting?
25. Does more training data always help? Why or why not?
26. What's the difference between L1 and L2 Regularization?
27. What happens if regularization strength (λ) is too high?
28. Why is a single train/test split sometimes unreliable?
29. How does K-Fold Cross Validation work?
30. When would you use Stratified K-Fold?
31. Why can't you use Linear Regression for classification?
32. What does the Sigmoid function do, and why is it needed?
33. What cost function does Logistic Regression use, and why not MSE?
34. Why is accuracy sometimes a misleading metric?
35. What's the difference between Precision and Recall?
36. Why does the F1 Score use a harmonic mean?
37. What does AUC represent?
38. When should you prefer Precision-Recall Curve over ROC?
39. Why does accuracy fail on imbalanced datasets?
40. What is SMOTE, and why is it preferred over naive over-sampling?
41. Why does KNN require feature scaling?
42. What is the Curse of Dimensionality?
43. How does a Decision Tree decide what to split on?
44. Why is a single Decision Tree considered unstable?
45. How does Random Forest reduce variance compared to a single tree?
46. Does increasing n_estimators cause Random Forest to overfit?
47. How does Boosting differ from Random Forest (bias vs variance, parallel vs sequential)?
48. What is the learning_rate / n_estimators tradeoff in Boosting?
49. Why does PCA require feature scaling, and what does "explained variance" mean?
50. Walk me through how you would approach an end-to-end ML project.