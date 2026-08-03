# Advanced Machine Learning Interview Questions

## Advanced Dimensionality Reduction

1. What problem are you solving when you apply dimensionality reduction in a production pipeline?
2. When would you apply dimensionality reduction for **model performance** vs **visualization**?
3. How do you decide the target number of dimensions for a dimensionality reduction step?
4. What are the most common ways dimensionality reduction can introduce **information leakage**?
5. How would you validate that dimensionality reduction is not harming downstream model performance?
6. You reduced dimensions and your model performance improved offline but worsened in production—what do you investigate?
7. How do you monitor a dimensionality reduction component in production (data drift / embedding drift)?
8. What are the tradeoffs between preserving global structure vs local neighborhoods in reduced spaces?
9. How would you version and roll out a dimensionality reduction change safely?
10. When is it better to skip dimensionality reduction and invest in better feature engineering instead?
11. How does dimensionality reduction affect interpretability and debugging?
12. In what cases does dimensionality reduction help with latency or memory constraints at inference?
13. How do you ensure train/validation/test transformations remain consistent over time?

---

## t-SNE

1. What is t-SNE typically used for in ML workflows?
2. What does t-SNE prioritize preserving in the embedding space?
3. Why is t-SNE commonly considered unsuitable as a general-purpose preprocessing step for supervised learning?
4. What is the practical meaning of the t-SNE `perplexity` parameter?
5. How do `perplexity` and dataset size interact in practice?
6. What symptoms indicate your t-SNE embedding is unstable or misleading?
7. How would you explain why distances between far-apart clusters in t-SNE are not reliable?
8. Your t-SNE plot shows “beautiful clusters” but clustering metrics are poor—what might be happening?
9. How do random seeds impact t-SNE results and how would you handle this in an analysis/report?
10. What are the most common misuse patterns of t-SNE in interviews and in real projects?
11. How would you scale t-SNE to larger datasets without freezing your laptop?
12. When would you choose t-SNE over UMAP?

---

## UMAP

1. What is UMAP typically used for compared to t-SNE?
2. What does UMAP’s `n_neighbors` control in practical terms?
3. What does `min_dist` change visually and behaviorally in the embedding?
4. When is UMAP a reasonable choice for generating reusable embeddings rather than just visualization?
5. How do you evaluate whether a UMAP embedding is stable enough to be used downstream?
6. You trained UMAP on one dataset version and the embedding behavior changed after refresh—what do you investigate?
7. How would you handle transform/new-point embedding in a pipeline using UMAP?
8. What are the risks of using UMAP embeddings as features for a supervised model?
9. When would you prefer PCA over UMAP even if the goal is dimensionality reduction?
10. How would you choose UMAP hyperparameters for a dataset with strong cluster structure vs continuous manifold structure?
11. What production monitoring would you add if UMAP embeddings are used as features?
12. When would you choose UMAP over t-SNE for stakeholder-facing visualizations?

---

## Clustering

1. What are the main reasons to use clustering in a real product (not a Kaggle exercise)?
2. How do you choose a clustering algorithm when you don’t know the number of clusters?
3. How do you evaluate clustering quality without labels?
4. What does “cluster stability” mean and how would you test it?
5. How do you prevent clustering from being dominated by a single scale-heavy feature?
6. Your clustering results change drastically after a minor data refresh—what do you do?
7. How do you handle outliers in clustering workflows?
8. How would you use clustering as a feature in a supervised model pipeline?
9. How would you present clustering results to stakeholders in a trustworthy way?
10. What are the most common causes of “clusters” that are artifacts of preprocessing?
11. You need clusters for downstream operational decisions—what are the risks and guardrails?
12. How do you choose between K-Means, GMM, and DBSCAN for a dataset with mixed densities?
13. What are common production failure modes for clustering systems?

---

## Gaussian Mixture Models (GMM)

1. What is the intuition behind GMM clustering compared to K-Means?
2. What does “soft assignment” mean and when is it useful?
3. How would you decide the number of components in a GMM in practice?
4. How does the choice of `covariance_type` affect clustering behavior?
5. When does a GMM fail even if you choose a reasonable number of components?
6. How would you use GMM cluster probabilities in downstream systems?
7. If your GMM assigns high probability to multiple clusters for many points, how do you interpret that?
8. What are practical signs that your GMM is overfitting?
9. How would you initialize and stabilize GMM training runs?
10. When would you choose a GMM over DBSCAN?
11. How would you handle outliers using a GMM-based approach?
12. What production constraints make GMM less suitable than simpler clustering methods?

---

## DBSCAN

1. What is DBSCAN’s core idea in plain engineering terms?
2. What does `eps` represent and why is it so sensitive?
3. What does `min_samples` control in DBSCAN?
4. How does DBSCAN handle noise and outliers?
5. In what scenarios is DBSCAN a strong choice over K-Means?
6. Why does DBSCAN struggle with varying density clusters?
7. How would you tune DBSCAN hyperparameters systematically without labels?
8. Your DBSCAN returns one giant cluster and many noise points—what do you try next?
9. Your DBSCAN returns many tiny clusters—what does that suggest about `eps` and `min_samples`?
10. How would you scale DBSCAN to large datasets?
11. How do you validate whether DBSCAN’s “noise” points are meaningful anomalies?
12. When should you avoid DBSCAN entirely?

---

## Anomaly Detection

1. What’s the difference between anomaly detection and imbalanced classification in practical terms?
2. How do you decide whether to solve the problem as supervised vs unsupervised anomaly detection?
3. What is the operational meaning of an “anomaly score”?
4. How would you select thresholds when you have limited investigation capacity?
5. What are the risks of training anomaly detectors on data that already contains anomalies?
6. How does feature scaling affect LOF and One-Class SVM?
7. When is Isolation Forest a strong baseline, and when is it likely to fail?
8. How do you evaluate anomaly detection without labels?
9. How do you evaluate anomaly detection with delayed labels?
10. What drift patterns are common in anomaly detection systems and how do you handle them?
11. Your anomaly alert volume spikes overnight—what do you investigate first?
12. How would you build a safe fallback when an anomaly model is down or unstable?
13. How do you reduce false positives without missing important anomalies?
14. How do you monitor anomaly detection systems in production (without immediate labels)?

---

## Feature Engineering

1. What’s the difference between feature engineering and feature extraction in practice?
2. What are the most common sources of data leakage in feature engineering?
3. How do you enforce training/serving parity for features?
4. How do you decide which features to build first for a new business problem?
5. What are “time-aware” features and how do they cause leakage if done incorrectly?
6. How do you handle categorical variables for boosting models vs linear models?
7. What is the engineering tradeoff between building complex features vs using a more powerful model?
8. How do you version features and prevent silent changes from breaking models?
9. You added new features and offline metrics improved but production got worse—what are your hypotheses?
10. How do you build robust features that survive drift and product changes?
11. What checks would you add to catch feature pipeline bugs early?
12. How would you debug a feature that suddenly becomes mostly null?

---

## ML Workflow

1. Walk through your end-to-end workflow for building an ML model for a real product.
2. What baselines do you build first and why?
3. How do you choose evaluation metrics aligned with business goals?
4. How do you ensure your offline evaluation matches production reality?
5. What is the difference between model evaluation and error analysis?
6. How do you decide when to stop iterating on a model and ship?
7. How do you design a safe rollout strategy for a new model?
8. What artifacts do you log for reproducibility and debugging?
9. How do you handle delayed labels in your workflow?
10. How do you incorporate monitoring and retraining planning into the initial design?
11. What is your approach to slice-based evaluation and why does it matter?
12. How do you prevent “test set overfitting” over multiple iterations?

---

## Hyperparameter Tuning

1. When is hyperparameter tuning worth the time vs focusing on data/features?
2. How do you tune hyperparameters without leaking information from the test set?
3. What’s the difference between random search and grid search in practice?
4. How do you define a tuning objective for imbalanced classification?
5. How do you incorporate time-based validation into a tuning workflow?
6. What’s your strategy for selecting the search space and budgets?
7. How do you detect that tuning is overfitting your validation set?
8. What are the highest-impact hyperparameters for boosting models?
9. What are the highest-impact hyperparameters for DBSCAN / UMAP / t-SNE?
10. How do you track experiments and ensure results are reproducible?
11. How do you tune decision thresholds separately from model training?
12. In production, how do you decide whether a tuned model is safe to roll out?

---

## Model Explainability

1. Why do production teams need explainability even when models are accurate?
2. What’s the difference between global and local explanations?
3. When would you use permutation importance vs SHAP?
4. What are common pitfalls of permutation importance?
5. What are common pitfalls of SHAP in real datasets?
6. How would correlated features affect your interpretation of feature importance?
7. How do you produce “reason codes” for an operational workflow?
8. How do you use explainability tools for debugging model regressions?
9. How do you evaluate whether explanations are stable across model retrains?
10. When can explainability be misleading or harmful?
11. How do you communicate model explanations to non-technical stakeholders?
12. What are the risks of claiming explanations imply causality?
13. You see that the model relies heavily on a suspicious feature—what steps do you take?

---

## Gradient Boosting Libraries (XGBoost / LightGBM / CatBoost)

1. Why do gradient boosted trees often dominate tabular datasets?
2. When would you choose XGBoost vs LightGBM vs CatBoost?
3. What makes CatBoost especially strong with categorical features?
4. What are the key hyperparameters you tune first for boosted trees?
5. How do you recognize overfitting in boosted tree training?
6. Your boosting model performs well offline but poorly in production—what do you check first?
7. How do you handle missing values and outliers in boosting pipelines?
8. How do you incorporate class imbalance handling for boosting models?
9. What are production considerations for model size and inference latency?
10. How do you add explainability to boosted tree predictions in production?
11. How would you tune boosting models with time-based validation?
12. What are common reasons LightGBM can overfit and how do you mitigate?
13. How do you decide whether a simpler model is preferable to boosted trees?

---

## Recommendation Systems

1. Compare popularity-based, content-based, and collaborative filtering recommenders.
2. Why is popularity baseline still important in mature recommender systems?
3. What’s the difference between candidate generation and ranking?
4. When would you choose item-based CF over user-based CF?
5. Explain the cold start problem for users and items and how you handle each.
6. Why is RMSE often a bad metric for recommender systems?
7. Define Precision@K and Recall@K; when do you prefer each?
8. Explain NDCG at a high level; why is rank position discounted?
9. How would you evaluate a recommender offline without leakage?
10. How do feedback loops happen in recommender systems and how do you mitigate them?
11. How do you incorporate diversity and freshness constraints without destroying relevance?
12. What monitoring would you set up for a recommender system?
13. How would you design a fallback strategy when personalization is unavailable?
14. Describe matrix factorization intuitively and what limitations it has in production.
15. What are embeddings in recommendation and why are they useful?

---

## Time Series

1. Why can’t time series data be treated like standard tabular IID data?
2. Explain trend vs seasonality vs noise with real examples.
3. What is stationarity in practical terms and why does it matter?
4. Why is random split wrong for time series forecasting?
5. What is walk-forward validation and when should you use it?
6. Compare moving average vs exponential smoothing vs ARIMA at a high level.
7. How do rolling features introduce leakage and how do you prevent it?
8. How do you choose the forecast horizon and why does it matter?
9. Compare MAE vs RMSE; when do you prefer each?
10. Why can MAPE be dangerous and how do you handle zeros?
11. How would you build a strong baseline for weekly-seasonal daily data?
12. What monitoring would you implement for a deployed forecasting model?
13. How do you detect structural breaks and what do you do operationally?
14. How do you decide retraining cadence for a forecasting model?
15. What are common causes of forecasting failures that are not “model issues”?

---

## Data Drift

1. Why can a model with strong validation metrics fail in production months later?
2. Define data drift, concept drift, and label drift.
3. Give examples of data drift caused by feature pipeline changes.
4. Give examples of concept drift in adversarial settings like fraud.
5. How would you detect drift without labels?
6. How would you detect drift with delayed labels?
7. What is the difference between monitoring feature distributions and monitoring prediction distributions?
8. How do you prioritize drift alerts when you have hundreds of features?
9. What does a sudden drift vs gradual drift vs recurring drift suggest operationally?
10. How do you respond when drift is detected: rollback vs retrain vs threshold adjustment?
11. How do you prevent retraining on corrupted or biased data?
12. What monitoring slices would you track and why?
13. How do feedback loops create drift in recommenders?
14. What runbooks would you write for common drift-related alerts?
15. How do you evaluate whether retraining actually fixed the drift problem?

---

## Feature Extraction vs Representation Learning

1. Distinguish feature engineering vs feature extraction vs representation learning.
2. Why did deep learning replace many classical feature-extraction pipelines for unstructured data?
3. When does classical ML still win in production and why?
4. Why do gradient boosted trees often beat neural networks on tabular data?
5. What changes when you move from sklearn’s fit/predict workflow to PyTorch training loops?
6. What stays the same when moving from classical ML to deep learning (engineering perspective)?
7. What are the operational costs of representation learning systems?
8. When would TF-IDF + linear model be a better choice than representation learning?
9. What are common training/serving skew issues when representations are learned?
10. How do you monitor drift when your features are embeddings?
11. What does “end-to-end learning” mean and what are the tradeoffs?
12. You have a small dataset with complex inputs—how would you choose between manual features and learned representations?
13. How do you justify deep learning complexity to a product team?
14. What is a safe migration plan from a classical model to a representation-learning-based system?