Machine Learning Model Selection Guide

1) Complete ML Decision Flowchart


flowchart TD

  A["Start: What problem are you solving?"] --> B{"Problem type"}

  B -->|Tabular supervised| C{"Dataset size & constraints"}

  B -->|Clustering| D{"Need #clusters k?"}

  B -->|DR or visualization| E{"Goal"}

  B -->|Anomaly detection| F{"Labels available?"}

  B -->|Recommendation| G{"Interaction history?"}

  B -->|Time series| H{"Forecasting horizon & seasonality"}

  %% Tabular supervised

  C -->|Strong baseline| C1["XGBoost / LightGBM / CatBoost"]

  C -->|Many categoricals| C2["CatBoost"]

  C -->|Large and fast| C3["LightGBM"]

  C -->|Mature ecosystem| C4["XGBoost"]

  C -->|Interpretability| C5["Logistic/Linear + constraints OR GBDT + explainability"]

  %% Clustering

  D -->|Yes| D1{"Overlapping / elliptical clusters?"}

  D -->|No| D2{"Expect noise & arbitrary shapes?"}

  D1 -->|Yes| D3["GMM"]

  D1 -->|No| D4["K-Means baseline"]

  D2 -->|Yes| D5["DBSCAN"]

  D2 -->|No| D6["Estimate k then K-Means/GMM"]

  %% Dimensionality reduction

  E -->|Compression| E1["PCA"]

  E -->|Visualization| E2{"Need stable embedding / transform new points?"}

  E2 -->|Yes| E3["UMAP"]

  E2 -->|No| E4["t-SNE"]

  %% Anomaly detection

  F -->|Yes| F1["Supervised classifier + threshold policy"]

  F -->|No| F2{"General robust tabular baseline?"}

  F2 -->|Yes| F3["Isolation Forest"]

  F2 -->|No| F4{"Local density anomalies?"}

  F4 -->|Yes| F5["LOF"]

  F4 -->|No| F6["One-Class SVM (small/medium) or rules baseline"]

  %% Recommenders

  G -->|No| G1["Popularity + Content-based + onboarding"]

  G -->|Yes| G2{"Good item metadata?"}

  G2 -->|Yes| G3["Hybrid: Content + CF"]

  G2 -->|No| G4["Collaborative filtering / MF embeddings"]

  G3 --> G5["Add retrieval + ranking + constraints as scale grows"]

  G4 --> G5

  %% Time series

  H --> I["Chronological split + walk-forward validation"]

  I --> J{"Strong seasonality?"}

  J -->|Yes| K["Seasonal naive → Exponential smoothing (Holt-Winters)"]

  J -->|No| L["Naive/Moving avg → Exponential smoothing"]

  K --> M{"Need stronger classical?"}

  L --> M

  M -->|Yes| N["ARIMA/SARIMAX"]

  M -->|No| O["Ship baseline + monitor drift"]

  N --> P{"Need exogenous drivers / many series?"}

  P -->|Yes| Q["ML with lags + calendar + exog"]

  P -->|No| R["Classical is enough"]


2) Comparison Tables

2.1 PCA vs t-SNE vs UMAP

| Dimension | PCA | t-SNE | UMAP |

|---|---|---|---|

| Primary purpose | compression, denoising | visualization | visualization (+ sometimes reusable embeddings) |

| Preserves | global variance | local neighborhoods | local + some global |

| Transform new points | Yes | Not reliably (often refit) | Yes (common) |

| Scalability | High | Medium/low | Medium/high |

| Interpretability | Medium (components) | Low | Low |

| Production suitability | High (as preprocessing) | Low | Medium (carefully) |

| Common wrong use | thinking PCA “finds clusters” | using for preprocessing or metrics | assuming embedding stability across retrains |

| Interview importance | High | Medium | Medium |

**Production default**

- **PCA** for compression/speed/denoising.

- **UMAP** for visualization; consider for embeddings only if you can version + monitor.

- **t-SNE** for exploratory visualization only.

2.2 K-Means vs GMM vs DBSCAN

| Dimension | K-Means | GMM | DBSCAN |

|---|---|---|---|

| Cluster shape | spherical-ish | elliptical | arbitrary shapes |

| Needs `k` | Yes | Yes | No |

| Handles outliers/noise | Poor | Medium | Strong (noise label) |

| Output type | hard assignment | soft probabilities | hard + noise |

| Scalability | High | Medium | Medium |

| Key hyperparameters | `n_clusters`, init | `n_components`, `covariance_type` | `eps`, `min_samples` |

| Production suitability | High baseline | Medium | Medium (sensitive tuning) |

| Interpretability | Medium | Medium | Medium |

| Interview importance | High | High | High |

**Production default**

- Start with **K-Means** as baseline if `k` is known and clusters are blob-like.

- Use **GMM** if overlap/soft membership matters.

- Use **DBSCAN** when noise/outliers are expected and density clusters are meaningful (but validate `eps` sensitivity).

2.3 Isolation Forest vs LOF vs One-Class SVM

| Dimension | Isolation Forest | LOF | One-Class SVM |

|---|---|---|---|

| Best for | general tabular anomalies | local density anomalies | boundary of normal |

| Scaling | Good | Medium | Poor–Medium |

| Training speed | Fast | Medium | Slow on big data |

| Key hyperparameters | `n_estimators`, `contamination`, `max_samples` | `n_neighbors`, `contamination` | `nu`, `kernel`, `gamma` |

| Interpretability | Low–Medium | Low | Low |

| Production suitability | High baseline | Medium | Low–Medium (fragile) |

| Typical failure | wrong contamination/threshold | sensitive to scale + neighbors | heavy tuning + compute |

| Interview importance | High | Medium | Medium |

**Production default**

- **Isolation Forest** as first unsupervised anomaly baseline for tabular.

- **LOF** when “local neighborhood” anomalies matter.

- **One-Class SVM** only for smaller datasets or when you’ve proven it wins.

2.4 XGBoost vs LightGBM vs CatBoost (tabular supervised)

| Dimension | XGBoost | LightGBM | CatBoost |

|---|---|---|---|

| Strength | robust default, mature | very fast, scalable | excellent categorical handling |

| Categorical features | needs encoding | supports (depends) | native + strong |

| Speed | fast | very fast | fast |

| Overfit risk | manageable | higher if unconstrained | manageable |

| Production suitability | very high | high | high |

| Typical pick when | general purpose | large datasets | many categoricals + minimal preprocessing |

| Key hyperparameters | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | `num_leaves`, `max_depth`, `learning_rate`, `min_data_in_leaf`, `feature_fraction` | `depth`, `learning_rate`, `l2_leaf_reg`, `iterations` |

| Interpretability | medium (SHAP works well) | medium | medium |

| Interview importance | High | High | High |

**Production default**

- **CatBoost** if categoricals are central and you want strong out-of-the-box performance.

- **LightGBM** if throughput/training speed at scale is the bottleneck.

- **XGBoost** if you want the safest all-around default and wide ecosystem support.

2.5 Popularity vs Content-Based vs Collaborative Filtering (Recommenders)

| Dimension | Popularity | Content-Based | Collaborative Filtering |

|---|---|---|---|

| Personalization | None | Medium | High |

| Needs interactions | No | Optional | Yes |

| Cold start (user) | Good | Medium | Poor |

| Cold start (item) | Medium | Good | Poor |

| Explainability | High | High | Medium |

| Production role | baseline/fallback/trending | new items + explainability | personalization core |

| Key failure mode | popularity bias | overspecialization | cold start + feedback loops |

| Interview importance | High | High | High |

**Production default**

- Always ship **popularity** baseline.

- Add **content-based** for item cold start and explainability.

- Add **CF/MF embeddings** once you have interaction volume.

- Most real systems become **hybrid**.

2.6 Forecasting: Classical vs ML vs Deep Learning

| Dimension | Classical (ES/ARIMA) | ML (GBDT/linear w/ features) | Deep Learning (future topic) |

|---|---|---|---|

| Data needs | low–medium | medium | high |

| Handles exogenous variables | limited (SARIMAX helps) | strong | strong |

| Many series at scale | challenging per-series tuning | strong (shared patterns) | strong (infra-heavy) |

| Interpretability | high–medium | medium | low–medium |

| Production complexity | low | medium | high |

| Best first choice | yes for univariate & baselines | yes for feature-rich demand | only when justified by scale |

| Interview importance | High | High | Medium |

**Production default**

- Start with **seasonal naive** + **exponential smoothing**.

- Use **ARIMA/SARIMAX** for strong univariate baseline.

- Prefer **ML with lags + calendar + exog** when drivers matter or you have many series.

- Deep learning only when data + infra justify it (not covered here beyond mention).

2.7 Manual Feature Engineering vs Representation Learning

| Dimension | Manual feature engineering | Representation learning |

|---|---|

| Best for | tabular business problems | unstructured data (text/image/audio), embeddings |

| Data requirement | lower | higher |

| Infra complexity | low–medium | medium–high |

| Interpretability | higher | lower |

| Latency | often lower | varies |

| Production risk | lower | higher (coupling, drift) |

| Interview framing | “strong baseline + good features” | “needed when features are hard to design” |

**Production default**

- Tabular: manual features + boosting.

- Unstructured: representation learning (or feature extraction baseline first).

3) Engineering Rules (50+ “If → Use → Reason”)

1. If you need **2D visualization** of high-dimensional data → Use **UMAP** → Better global structure and reusable transform than t-SNE.

2. If you need **quick sanity visualization only** → Use **t-SNE** → Great local separation for exploration.

3. If you need **compression for modeling** → Use **PCA** → Fast, stable, transformable, production-friendly.

4. If you plan to **embed new points later** → Use **UMAP/PCA**, not t-SNE → Transform support + stability.

5. If you see “clusters” only after many t-SNE reruns → Don’t trust it → Instability is a red flag.

6. If `k` is known and clusters are blob-like → Use **K-Means** → Strong baseline and scalable.

7. If you need **soft cluster membership** → Use **GMM** → Probabilistic assignments.

8. If clusters overlap and are elliptical → Use **GMM** → Matches geometry better than K-Means.

9. If you expect **noise/outliers** and arbitrary shapes → Use **DBSCAN** → Explicit noise handling.

10. If densities vary widely across clusters → Avoid **DBSCAN** → Single `eps` won’t fit all.

11. If you don’t know `k` but data is clean and small → Use **hierarchical clustering** → Helps explore structure.

12. If you need a robust unsupervised anomaly baseline on tabular → Use **Isolation Forest** → Strong default with decent scaling.

13. If anomalies are “locally weird” in neighborhoods → Use **LOF** → Detects local density drops.

14. If dataset is large and you’re considering One-Class SVM → Don’t → Scaling and tuning pain.

15. If labels exist for anomalies → Use **supervised classification** → Beats unsupervised when labels are reliable.

16. If your anomaly threshold must match review capacity → Choose threshold by capacity → Operations > metrics.

17. If your data is tabular and you want best performance quickly → Use **GBDT** → Strong default.

18. If you have many categoricals and want minimal encoding → Use **CatBoost** → Native categorical strength.

19. If you have massive dataset and training speed is a bottleneck → Use **LightGBM** → Very fast.

20. If you want safest ecosystem and reproducibility → Use **XGBoost** → Mature tooling.

21. If a model wins offline but is too slow online → Don’t ship it → Latency is a requirement.

22. If your evaluation uses random split on time data → Fix split → It’s likely leakage.

23. If you build rolling features → `shift(1)` before rolling → Prevent leakage.

24. If you compute aggregates per entity (user/card) → Compute past-only windows → Prevent future leakage.

25. If model performance varies by region/device → Add slice monitoring → Aggregate metrics hide failures.

26. If feature null rate spikes → Assume pipeline break first → Most common prod failure.

27. If score distribution shifts sharply → Investigate drift → Can indicate upstream changes.

28. If labels are delayed → Monitor proxies (scores, decision rates) → Don’t wait weeks to detect issues.

29. If base rate changes (label drift) → Recalibrate/adjust thresholds → Retraining may not be needed.

30. If concept drift is suspected (performance drop with stable inputs) → Retrain on recent data → Relationship changed.

31. If drift is sudden and broad → Check logging/schema changes → Often not “model went bad”.

32. If recommender has no interaction data for a user → Use **popularity + onboarding** → User cold start.

33. If recommender has new items → Use **content-based + exploration** → Item cold start.

34. If recommender needs scale → Use **embeddings + ANN retrieval** → Efficient candidate generation.

35. If recommender UI shows top K items → Optimize **NDCG/Precision@K** → Ranking not RMSE.

36. If evaluating recommenders offline → Use time-aware splits → Avoid future leakage.

37. If you can’t log impressions → Expect biased offline evaluation → Exposure bias.

38. If you need forecasting now → Start with **seasonal naive** → Strong baseline, hard to beat.

39. If series has trend + seasonality → Use **Holt-Winters** → Practical robust classical method.

40. If you need stronger univariate baseline → Use **SARIMA/SARIMAX** → Good classical upgrade.

41. If forecasting includes promos/weather → Prefer **ML with exog** → Classical alone may miss drivers.

42. If MAPE is requested but there are zeros → Refuse or modify → MAPE explodes.

43. If big forecast errors are costly → Prefer **RMSE** → Penalizes large misses.

44. If you want robust forecast comparison → Prefer **MAE** → Less outlier sensitive.

45. If you’re tempted to use deep learning for small tabular → Don’t → Boosting usually wins.

46. If you need explainability for compliance → Prefer simpler models or strong explainability tooling → Operational requirement.

47. If features are highly correlated → Be cautious with permutation importance/PDP → Misleading attributions.

48. If you need local explanations per decision → Use SHAP (budget for cost) → Reason codes.

49. If you deploy a new model → Canary/shadow first → Reduce blast radius.

50. If you retrain frequently → Ensure data quality gates → Avoid learning corrupted data.

51. If monitoring alerts fire often → Add runbooks + thresholds tuning → Avoid alert fatigue.

52. If your clustering isn’t stable across runs → Treat as exploratory only → Not production insight yet.

53. If your UMAP embedding changes across retrains → Version embeddings and revalidate downstream → Stability isn’t guaranteed.

54. If you have strict memory constraints → Avoid huge dense embeddings for everything → Use sparse/ANN strategies.

55. If you need the fastest “what should I use” on tabular classification → **CatBoost if many categoricals; else XGBoost/LightGBM** → Defaults.

4) Production Recommendations (What to choose and why)

4.1 Practical defaults by problem type

| Problem type | Production-first default | Why |

|---|---|---|

| Tabular classification/regression | **GBDT** (XGBoost/LightGBM/CatBoost) | strong, robust, fast iteration |

| Clustering baseline | **K-Means** (if `k` known) | scalable, simple |

| Overlapping clusters | **GMM** | soft membership helps |

| Clustering with noise/outliers | **DBSCAN** (after sensitivity checks) | explicit noise |

| Dimensionality reduction for modeling | **PCA** | stable transform |

| Visualization embedding | **UMAP** | better reuse and speed |

| Unsupervised anomaly detection | **Isolation Forest** | best general baseline |

| Recommendation cold start | **Popularity + content** | immediate value + coverage |

| Recommendation personalization | **Hybrid (CF + content + popularity)** | robust, production standard |

| Forecasting baseline | **Seasonal naive + exponential smoothing** | fast + strong baseline |

| Forecasting with exog drivers | **ML with lags + calendar + exog** | captures drivers; scales across series |

4.2 What “production suitability” really means

Before choosing an algorithm, answer:

- Can we compute the required features **online**?

- Can we meet **latency** and **throughput**?

- Do we have a plan for **monitoring**, **drift**, and **retraining**?

- Do we need **interpretability** or **reason codes**?

- Can we version artifacts and reproduce training?

If any answer is “no”, prefer the simpler approach.

5) Common Wrong Choices

1. Using **t-SNE** as a preprocessing step for modeling (unstable, not designed for transform).

2. Treating a **t-SNE/UMAP plot** as proof that clusters are real.

3. Using **DBSCAN** on data with **varying densities** and expecting clean clusters.

4. Using **One-Class SVM** on large datasets without profiling runtime.

5. Using **MAPE** on data with zeros/near-zeros and trusting the result.

6. Using **random train/test split** for time series or time-dependent logs.

7. Evaluating recommenders with **RMSE** when the product is ranking top-K.

8. Building collaborative filtering without a plan for **cold start**.

9. Training a deep model on **tiny tabular** dataset because “deep is better”.

10. Choosing a model without considering **serving constraints** (latency/memory).

11. Retraining automatically on drift alerts without verifying data quality (learning on broken data).

12. Deploying without monitoring prediction distributions or null rates.

13. Assuming SHAP explanations are causal truth.

14. Not versioning preprocessing—training/serving skew silently kills performance.

6) Interview Decision Questions (Scenario-based only)

1. You need a 2D visualization of embeddings for a presentation. Which method and why?

2. You need an embedding method that can transform new samples later. Which method and why?

3. You have blob-like clusters and you know `k`. What do you start with?

4. You don’t know `k` and expect lots of outliers. What clustering method and what hyperparameters matter?

5. Your DBSCAN results change drastically with small `eps` changes—what does that imply and what do you do?

6. You need soft assignment for customer segments to support targeted marketing. Which clustering model fits?

7. You have no labels and need anomaly detection for tabular sensor data; must run daily on CPU. What do you choose?

8. You suspect anomalies are local to neighborhoods (contextual). What do you choose and what can go wrong?

9. You have many categorical variables and want minimal preprocessing for a classification model. Which booster?

10. You need the fastest training over tens of millions of rows. Which booster and what constraints do you add to avoid overfit?

11. You’re building a recommender for a brand-new product with no interaction data. What do you launch with?

12. You have interaction logs and a large catalog; personalization matters. What recommender architecture do you propose?

13. Your recommender model offline RMSE improved but CTR dropped online. What’s your diagnosis?

14. You must forecast daily sales with strong weekly seasonality. What baseline do you use?

15. You must forecast demand where promotions drive most variation. Do you choose classical methods or ML, and why?

16. Your time series evaluation used random split and looks great. What do you say?

17. Your model’s validation metrics are strong but production performance drops after two months. What drift types might be happening?

18. You see null rates spike in a critical feature. What’s your first response?

19. Labels arrive after 2 weeks. What monitoring do you put in place to detect issues sooner?

20. You need per-decision explanations for a risk scoring model. What explainability approach do you propose?

7) One-page Cheat Sheet

Quick picks

- **Tabular supervised:** GBDT → CatBoost (many categoricals), LightGBM (scale/speed), XGBoost (safe default).

- **Dimensionality reduction:**

- PCA = preprocessing/compression

- UMAP = visualization + reusable transform

- t-SNE = visualization only

- **Clustering:**

- K-Means = fast baseline, needs k

- GMM = overlapping/elliptical + soft assignments

- DBSCAN = arbitrary shapes + noise (watch `eps`)

- **Anomaly detection:**

- Isolation Forest = default baseline

- LOF = local density anomalies

- One-Class SVM = small data / special cases

- **Recommenders:**

- Start: popularity + content

- Mature: hybrid (CF + content + popularity), retrieval + ranking

- **Forecasting:**

- Start: seasonal naive + exponential smoothing

- Strong univariate: SARIMA/SARIMAX

- Drivers/many series: ML with lags + calendar + exog

- **Feature strategy:**

- Tabular: manual features + boosting

- Unstructured: representation learning (or feature extraction baseline first)

High-impact hyperparameters to remember

- t-SNE: `perplexity`, `learning_rate`, `n_iter`

- UMAP: `n_neighbors`, `min_dist`, `n_components`

- DBSCAN: `eps`, `min_samples`

- GMM: `n_components`, `covariance_type`

- Isolation Forest: `contamination`, `n_estimators`, `max_samples`

- LightGBM: `num_leaves`, `min_data_in_leaf`, `learning_rate`

- XGBoost: `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, `learning_rate`

- CatBoost: `depth`, `iterations`, `learning_rate`, `l2_leaf_reg`

Fast red flags

- Random split on time data → leakage.

- t-SNE used for preprocessing → wrong tool.

- DBSCAN on varying density → unstable clustering.

- MAPE with zeros → nonsense.

- “Accuracy” used for fraud/anomaly → misleading.

Production mindset

- Baselines first.

- Correct splits.

- Version features + models.

- Monitor data quality + prediction distribution + delayed labels.

- Have rollback + retrain plan.