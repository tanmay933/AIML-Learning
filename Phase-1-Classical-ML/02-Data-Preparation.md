# Data Preparation

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain why data quality determines model quality more than algorithm choice
- Identify data types and know how each shapes preprocessing decisions
- Handle missing values, duplicates, and outliers with sound judgment
- Apply feature scaling and categorical encoding correctly, and know when each is needed
- Perform feature engineering that improves model performance
- Recognize and prevent data leakage — one of the most costly real-world ML mistakes
- Follow a professional, correctly-ordered data preparation workflow
- Answer interview questions on preprocessing with engineering-grade explanations

---

# Why This Topic Exists

Every ML algorithm you'll learn in this handbook — Linear Regression, Decision Trees, KMeans, PCA — assumes the data it receives is already clean, numeric, and well-structured. Real-world data never arrives that way.

Data Preparation is the bridge between messy, raw data and a dataset a model can actually learn from. Skipping or rushing this step is the single most common reason ML projects fail in practice — not because the algorithm was wrong, but because the data feeding it was flawed.

This chapter builds the permanent preprocessing toolkit you'll reuse in every chapter that follows.

---

# Intuition

💡 **Intuition**: A model can only be as good as the patterns present in its training data. If the data is noisy, inconsistent, or misleading, the model will faithfully learn those flaws too.

This is best captured by a phrase you'll hear constantly in industry:

> **"Garbage In, Garbage Out."**

⭐ **Must Know**: No algorithm — no matter how advanced — can compensate for fundamentally poor data. A state-of-the-art model trained on bad data will reliably produce bad predictions, often *confidently* bad, which is worse than obviously bad.

**Practical examples:**

- A house price model trained on data where 30% of "Area" values are missing or wrongly entered as `0` will learn a distorted relationship between area and price.
- A fraud detection model trained on duplicated fraud cases will overestimate how common fraud actually is.
- A churn model where categorical fields like `"USA"`, `"U.S.A"`, and `"United States"` are treated as three different countries will fragment a real signal into noise.

🚀 **Practical Insight**: In real ML teams, engineers typically spend **60–80% of project time** on data understanding and preparation, and only a small fraction on model selection and tuning. This ratio surprises most beginners, who expect the opposite.

---

# Core Concepts

## 1. The Real-World ML Pipeline

Data preparation isn't one step — it's a sequence of stages, each solving a specific problem with the raw data.

```
Raw Data
   ↓
Cleaning
   ↓
Feature Engineering
   ↓
Encoding
   ↓
Scaling
   ↓
Train/Test Split
   ↓
Model Training
```

| Stage | Purpose |
|---|---|
| **Raw Data** | Data exactly as collected — messy, inconsistent, possibly incomplete |
| **Cleaning** | Fix or remove missing values, duplicates, and problematic outliers |
| **Feature Engineering** | Create or transform features to better expose patterns to the model |
| **Encoding** | Convert categorical data into numeric form models can process |
| **Scaling** | Bring numeric features to comparable ranges |
| **Train/Test Split** | Separate data so model evaluation reflects real-world, unseen performance |
| **Model Training** | Fit the algorithm on the fully prepared training data |

⚠ **Common Mistake**: This exact order matters — especially where scaling happens relative to splitting. We'll return to this in the **Data Leakage** section, because getting this order wrong is one of the most common real-world bugs in ML pipelines.

---

## 2. Types of Data

Different data types require fundamentally different preprocessing. Recognizing type is always the first step.

| Type | Description | Example | Typical Preprocessing |
|---|---|---|---|
| **Numerical** | Continuous or discrete numbers | Age, Income, Area | Scaling, outlier handling |
| **Categorical** | Unordered categories | Color, City, Payment Method | Encoding |
| **Ordinal** | Categories with a meaningful order | Education Level (High School < Bachelor's < Master's) | Ordered encoding (order matters!) |
| **Binary** | Two possible values | Yes/No, True/False, Churned/Not | Simple 0/1 encoding |
| **Date/Time** | Timestamps | Signup Date, Transaction Time | Extract components (day of week, month, recency) |
| **Text** | Free-form text | Reviews, Emails | Not covered here — belongs to NLP (later phase) |

💡 **Intuition**: A model only understands numbers. Every preprocessing technique in this chapter ultimately exists to turn real-world data — whatever its original type — into clean, meaningful numbers.

📌 **Revision Point**: Ordinal ≠ Categorical. Ordinal data has an inherent order that must be *preserved* during encoding; treating it as unordered categorical data throws away useful information.

---

## 3. Missing Values

### What They Are and Why They Happen

Missing values are entries with no recorded data — often shown as `NaN`, `null`, empty strings, or placeholder values like `-1` or `"Unknown"`.

**Common causes:**

- A user skipped an optional form field
- A sensor failed to record a reading
- Data merged from multiple sources with inconsistent fields
- A field simply didn't apply to that record (e.g., "Spouse Name" for an unmarried person)

### Why They're Dangerous

⚠ **Common Mistake**: Ignoring missing values instead of explicitly handling them. Many ML algorithms will either throw errors or silently produce distorted results (e.g., treating `NaN` as `0`, which may be a completely wrong assumption).

⭐ **Must Know**: Missing values aren't always "random" — sometimes the fact that a value is missing is itself informative (e.g., missing income data might correlate with unemployment). Understand *why* data is missing before deciding how to handle it.

### Handling Strategies

| Strategy | How it Works | Pros | Cons | When to Use |
|---|---|---|---|---|
| **Drop rows** | Remove samples with missing values | Simple, no distortion of remaining data | Loses data; risky if many rows affected | Missing values are rare (<5%) and random |
| **Drop columns** | Remove entire feature | Simple; removes unreliable feature entirely | Loses potentially useful information | A feature is missing in the majority of rows |
| **Mean Imputation** | Fill with the column's average | Simple, preserves row count | Distorted by outliers; ignores relationships between features | Numerical data, roughly symmetric distribution, no major outliers |
| **Median Imputation** | Fill with the column's median | Robust to outliers | Still ignores relationships between features | Numerical data with skew or outliers |
| **Mode Imputation** | Fill with the most frequent value | Works for categorical data | Can artificially inflate the most common category | Categorical data |

🎯 **Interview Tip**: If asked "Mean vs Median imputation?" — the key differentiator is **outlier sensitivity**. Mean is pulled toward extreme values; median is not. Default to median when the feature is skewed or has outliers.

🚀 **Practical Insight**: Before choosing a strategy, always check *how much* data is missing and *whether it's missing at random*. A column missing 2% of values behaves very differently from one missing 60%.

---

## 4. Duplicate Data

💡 **Intuition**: A duplicate record is the same information counted more than once. If left in the dataset, the model effectively "hears" that data point multiple times, giving it artificially more influence than it deserves.

**Why duplicates bias models:**

- They skew the learned patterns toward over-represented samples
- They can cause identical rows to appear in both train and test sets, artificially inflating evaluation performance (a subtle form of data leakage)

**How duplicates are handled:**

- Identify duplicates using all columns (exact duplicate rows) or a subset of key columns (e.g., same `user_id` and `timestamp`)
- Typically dropped, keeping only the first occurrence — unless there's a legitimate reason multiple identical rows represent distinct real events

⚠ **Common Mistake**: Assuming duplicates only mean *identical rows*. Sometimes duplicates are near-identical due to formatting differences (e.g., `"john@email.com"` vs `"John@Email.com"`) — these require cleaning before duplicate detection will catch them.

---

## 5. Outliers

### What They Are

An outlier is a data point that differs significantly from the rest of the dataset.

**Why they occur:**

- Genuine rare events (a legitimate $10M house sale in a dataset of $200K–$500K homes)
- Data entry errors (age recorded as `250`)
- Sensor/measurement glitches
- Fraud or anomalous behavior (which, in some problems, is exactly what you're trying to detect)

### When Outliers Are Useful vs Harmful

| Situation | Treatment |
|---|---|
| Outlier represents a genuine rare event relevant to the problem | Keep it — it may be exactly what the model needs to learn |
| Outlier is a fraud/anomaly detection **target** | Definitely keep it — it's the signal, not noise |
| Outlier is a data entry error or sensor glitch | Remove or correct it |
| Outlier distorts a model sensitive to scale (e.g., Linear Regression) without adding real signal | Consider removing or transforming it |

⭐ **Must Know**: **Never remove outliers blindly.** Always apply domain knowledge first — ask "is this value physically/logically possible?" before deciding whether it's noise or a rare-but-real event.

### Detection Methods (Intuition Only)

**IQR (Interquartile Range) Method**

💡 **Intuition**: Look at where the "middle 50%" of your data lives, then flag anything unusually far outside that range as a potential outlier.

```
     Q1            Median            Q3
      |----------------|----------------|
      |<-- middle 50% of the data -->|

Outliers:  far below Q1  or  far above Q3
```

**Z-score Method**

💡 **Intuition**: Measures how many standard deviations a point is from the mean. A point very far from the mean (commonly, beyond ±3 standard deviations) is flagged as a potential outlier.

We're not deriving either formula here — just know the underlying idea: **both methods measure "how unusual is this point relative to the rest of the data?"**

🎯 **Interview Tip**: If asked to compare them — IQR is more robust to extreme values and skewed distributions; Z-score assumes data is roughly normally distributed. In practice, IQR is more commonly used as a first pass because it makes fewer assumptions about the data's shape.

---

## 6. Feature Scaling

### Why Scaling Exists

💡 **Intuition**: Imagine a dataset with `Age` (range 18–90) and `Income` (range 20,000–500,000). Some algorithms calculate distances or gradients using raw feature values — and Income, having much larger numbers, will completely dominate Age, even if Age is equally or more important. Scaling puts features on a comparable footing.

### Standardization vs Normalization

| Technique | What It Does | Resulting Range | Sensitive to Outliers? |
|---|---|---|---|
| **Standardization** (Z-score scaling) | Centers data around mean 0 with standard deviation 1 | No fixed range (typically roughly -3 to 3) | Less sensitive |
| **Normalization** (Min-Max Scaling) | Rescales data into a fixed range | Typically [0, 1] | Very sensitive (extreme values distort the range) |

💡 **Intuition**:
- **Standardization** answers: *"How many standard deviations away from average is this value?"*
- **Normalization** answers: *"Where does this value fall between the minimum and maximum?"*

🎯 **Interview Tip**: Standardization is generally the safer default in practice, especially when the data has outliers, because normalization compresses everything relative to the min/max — and a single extreme outlier can squash the rest of the data into a tiny sub-range.

### Which Algorithms Care About Scaling?

| Algorithm Category | Scaling Required? | Why |
|---|---|---|
| **KNN** | Yes | Relies directly on distances between points — unscaled features distort distance calculations |
| **Linear Regression** | Yes (recommended) | Coefficients and optimization are affected by feature magnitude |
| **Logistic Regression** | Yes (recommended) | Same reasoning — gradient-based optimization is sensitive to feature scale |
| **Decision Trees** | No | Trees split based on threshold comparisons per feature independently — scale doesn't affect split decisions |
| **Random Forest / Boosting** | No | Tree-based, inherits the same scale-insensitivity |

📌 **Revision Point**: If an algorithm relies on **distance** or **gradient-based optimization**, it almost certainly needs scaled features. If it relies on **threshold-based splitting** (trees), scaling is unnecessary.

We're not teaching these algorithms yet — just remember this table; it'll make immediate sense once you reach those chapters.

---

## 7. Encoding Categorical Variables

### Why Encoding Is Necessary

⭐ **Must Know**: ML models operate on numbers — they cannot directly interpret text categories like `"Red"`, `"Blue"`, `"Green"`. Encoding converts categories into a numeric representation the model can use.

### Label Encoding

Assigns each category an integer:

```
Red    → 0
Blue   → 1
Green  → 2
```

| Pros | Cons |
|---|---|
| Simple, memory-efficient | Implies a false numeric order/relationship (model may think Green > Blue > Red) |
| Works naturally for ordinal data | Misleading for purely categorical (nominal) data |

### One-Hot Encoding

Creates a separate binary column per category:

| Color_Red | Color_Blue | Color_Green |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

| Pros | Cons |
|---|---|
| No false ordering implied | Increases dimensionality significantly with many categories |
| Safe default for nominal (unordered) categories | Can create sparse, memory-heavy datasets |

### When to Use Each

| Data Type | Preferred Encoding |
|---|---|
| Ordinal (has meaningful order) | Label Encoding (with order explicitly preserved) |
| Nominal (no order), few categories | One-Hot Encoding |
| Nominal, high-cardinality (100s of categories) | Neither alone is ideal — often requires more advanced techniques (grouping rare categories, target encoding) — not covered in this handbook |

⚠ **Common Mistake**: Using Label Encoding on nominal (unordered) data like `City` or `Payment Method`. This silently introduces a fake numeric relationship the model will try to learn, hurting performance.

🎯 **Interview Tip**: If asked "One-Hot vs Label Encoding?" — the deciding factor is **whether the categories have a meaningful order**. Ordinal → Label Encoding. Nominal → One-Hot Encoding.

🚀 **Practical Insight**: High-cardinality categorical features (e.g., `Zip Code` with 40,000 unique values) are a real-world pain point — One-Hot Encoding would create 40,000 new columns. In practice, engineers group rare categories into `"Other"`, or use frequency/target encoding. We won't go deep here, but be aware this is a common interview follow-up question.

---

## 8. Feature Engineering

### What It Is

💡 **Intuition**: Feature engineering is the process of creating new, more informative features — or transforming existing ones — so the model can find patterns more easily.

⭐ **Must Know**: **Better features often beat better models.** A simple algorithm with well-engineered features frequently outperforms a complex algorithm fed raw, unprocessed data.

### Practical Examples

| Original Feature(s) | Engineered Feature | Why It Helps |
|---|---|---|
| `Age` | `Age Group` (e.g., 18–25, 26–35...) | Captures non-linear age effects that raw age might obscure |
| `Date` | `Day of Week` | Many behaviors (shopping, traffic) vary meaningfully by weekday vs weekend |
| `Income`, `Expenses` | `Savings Rate = (Income - Expenses) / Income` | Directly captures a financially meaningful ratio the model would otherwise have to "discover" indirectly |
| `Area`, `Bedrooms` | `Area per Bedroom = Area / Bedrooms` | A more direct signal of space quality than either feature alone |

### Feature Extraction vs Feature Creation

| Concept | Meaning | Example |
|---|---|---|
| **Feature Extraction** | Pulling a useful sub-piece out of an existing feature | Extracting `Day of Week` from a `Date` field |
| **Feature Creation** | Combining multiple features into a new, more meaningful one | `Savings Rate` from `Income` and `Expenses` |

🚀 **Practical Insight**: In real-world ML work, feature engineering is often where the most business value is created. Domain knowledge — understanding *what actually matters* in the problem — tends to produce better features than any automated technique.

---

## 9. Data Leakage

This is one of the most important — and most commonly violated — concepts in applied ML.

### What It Is

💡 **Intuition**: Data leakage happens when information that wouldn't be available at real prediction time accidentally "leaks" into the training process — making the model look far better than it actually is.

⭐ **Must Know**: A model suffering from data leakage will show excellent performance during development, then **fail in production**, because the leaked information simply won't exist when making real, live predictions.

### Common Causes

| Type of Leakage | Example | Why It's a Problem |
|---|---|---|
| **Scaling before splitting** | Computing mean/std for standardization using the *full* dataset, then splitting | The scaler has "seen" test data statistics, inflating test performance artificially |
| **Using future information** | Predicting churn using a feature like "Total Purchases in Next 6 Months" | This data wouldn't exist at prediction time in the real world |
| **Leakage through target variables** | Including a feature that's essentially a proxy for the target (e.g., predicting loan default using "Days Since Payment Missed") | The feature effectively already contains the answer |

### How to Avoid It

- ⭐ **Always split data before** computing any statistics used for preprocessing (mean, std, min/max, etc.)
- Fit scalers/encoders **only on the training set**, then apply that same transformation to the test set
- Carefully audit features for anything that wouldn't realistically be available at prediction time
- Ask for every feature: *"Would I actually have this information at the moment I need to make this prediction?"*

```
CORRECT ORDER:

Raw Data → Split (Train/Test) → Fit scaler on TRAIN only
              → Transform TRAIN and TEST using that same fitted scaler
```

⚠ **Common Mistake**: Fitting a `StandardScaler` (or any transformer) on the entire dataset before splitting. This is one of the most frequent real-world bugs — and one of the easiest to miss in code review.

🎯 **Interview Tip**: Data leakage questions are common in interviews because they test real engineering judgment, not just textbook knowledge. Always mention: *"Any transformation that learns statistics from data (scaling, imputation, encoding) must be fit only on the training set, then applied to the test set — never the reverse."*

---

## 10. Data Preparation Workflow

Putting everything together, here's the professional, correctly-ordered workflow:

```
Collect Data
   ↓
Understand Data          (EDA: distributions, missing values, types)
   ↓
Clean Data                (fix inconsistent formatting, invalid entries)
   ↓
Handle Missing Values
   ↓
Remove Duplicates
   ↓
Treat Outliers
   ↓
Split Data                 (train/test — BEFORE fitting scalers/encoders)
   ↓
Encode Categories          (fit on train, apply to test)
   ↓
Scale Features             (fit on train, apply to test)
   ↓
Feature Engineering        (can occur before/after split depending on the feature)
   ↓
Train Model
```

⭐ **Must Know**: Notice that **Split Data** happens *before* Encoding and Scaling in this professional workflow — this is the direct, practical application of the leakage-prevention principle from the previous section. (Chapter 1 introduced splitting conceptually; here you see exactly where it belongs in the real pipeline.)

📌 **Revision Point**: Feature engineering that only uses information from a single row (e.g., `Area per Bedroom`) is generally safe either before or after splitting. Feature engineering that involves aggregate statistics across rows (e.g., "average purchase amount per customer") must be computed carefully to avoid leaking test-set information into train-set features.

---

## 11. Common Beginner Mistakes

| Mistake | Consequence |
|---|---|
| **Scaling before splitting** | Data leakage — inflated, unrealistic test performance |
| **Encoding nominal data with Label Encoding** | Model learns a false ordinal relationship between categories |
| **Removing too much data** | Reduced training data, potential loss of important patterns |
| **Ignoring missing values** | Errors during training, or silently corrupted results |
| **Blindly removing all outliers** | Loss of genuinely important rare events (e.g., fraud cases) |
| **Forgetting feature engineering** | Model misses easily-available signal that raw features don't expose directly |
| **Not checking for duplicates** | Artificially inflated importance of repeated data points |

---

## 12. Interview Tips

**Q: Why is data preparation important?**
> Model quality is fundamentally limited by data quality — no algorithm can compensate for poor, inconsistent, or leaky data. In practice, most of the effort in a real ML project goes into preparing data correctly, not choosing the algorithm.

**Q: Mean vs Median Imputation?**
> Mean imputation fills missing values with the column average — simple, but sensitive to outliers. Median imputation uses the middle value and is more robust when data is skewed or contains outliers.

**Q: One-Hot Encoding vs Label Encoding?**
> Label Encoding assigns integers to categories and is appropriate for ordinal data with a natural order. One-Hot Encoding creates separate binary columns per category and is safer for nominal (unordered) data, since Label Encoding would falsely imply an order.

**Q: Standardization vs Normalization?**
> Standardization rescales data to have mean 0 and standard deviation 1, and is more robust to outliers. Normalization (Min-Max scaling) rescales data into a fixed range like [0,1], but is more sensitive to extreme values.

**Q: What is Data Leakage?**
> Data leakage occurs when information unavailable at real prediction time influences training, producing unrealistically good performance during development that doesn't hold up in production. Common cause: fitting scalers or encoders on the full dataset before splitting into train/test.

**Q: Why scale features at all?**
> Some algorithms (like KNN, Linear/Logistic Regression) rely on distance calculations or gradient-based optimization, which are sensitive to the relative magnitude of features. Without scaling, features with larger numeric ranges can dominate the model unfairly.

**Q: Do Decision Trees require feature scaling?**
> No. Trees split data based on threshold comparisons on individual features, which are unaffected by the feature's scale.

**Q: What are outliers, and should you always remove them?**
> Outliers are data points that differ significantly from the rest of the dataset. They should not always be removed — domain knowledge should determine whether an outlier is a data error (remove) or a genuine rare event relevant to the problem (keep).

**Q: What is Feature Engineering?**
> The process of creating new features or transforming existing ones to better expose meaningful patterns to a model. Well-engineered features often improve performance more than switching to a more complex algorithm.

---

# Quick Revision

## Preprocessing Checklist

- [ ] Understood data types (numerical, categorical, ordinal, binary, date/time)
- [ ] Identified and handled missing values appropriately
- [ ] Removed or investigated duplicate records
- [ ] Investigated outliers using domain knowledge before removing them
- [ ] Split data into train/test **before** fitting any scaler or encoder
- [ ] Encoded categorical variables using the correct method (ordinal vs nominal)
- [ ] Scaled numerical features if the target algorithm requires it
- [ ] Engineered features that add real, meaningful signal
- [ ] Verified no data leakage exists anywhere in the pipeline

## Comparison Tables Recap

| Missing Value Strategy | Best For |
|---|---|
| Drop rows | Small amount of random missingness |
| Drop columns | Feature missing in majority of rows |
| Mean Imputation | Symmetric numeric data, no outliers |
| Median Imputation | Skewed numeric data, has outliers |
| Mode Imputation | Categorical data |

| Scaling Method | Best For |
|---|---|
| Standardization | Default choice, robust to outliers |
| Normalization | Bounded ranges needed, data has few/no outliers |

| Encoding Method | Best For |
|---|---|
| Label Encoding | Ordinal data (order matters) |
| One-Hot Encoding | Nominal data (no order), few categories |

## Essential Terminology

- **Imputation** — filling in missing values
- **Outlier** — a data point far from the rest of the distribution
- **Standardization** — scale to mean 0, std 1
- **Normalization** — scale to a fixed range, typically [0,1]
- **Label Encoding** — categories mapped to integers
- **One-Hot Encoding** — categories mapped to binary columns
- **Feature Engineering** — creating/transforming features to improve model signal
- **Data Leakage** — unintended use of information unavailable at prediction time

## Interview Facts Cheat Sheet

- Data quality > algorithm complexity, almost always.
- Always split **before** fitting any preprocessing transformer.
- Trees don't need scaling; distance-based and gradient-based models do.
- Ordinal → Label Encoding. Nominal → One-Hot Encoding.
- Never blindly remove outliers — check domain relevance first.
- Data leakage produces performance that looks great in development and fails in production.

## ✅ Checklist: "I understand this chapter if I can..."

- [ ] Explain why data quality matters more than model choice, with an example
- [ ] Identify the data type of any given feature and state its typical preprocessing
- [ ] Choose an appropriate missing value strategy and justify it
- [ ] Explain why duplicates bias a model
- [ ] Decide whether an outlier should be kept or removed, using domain reasoning
- [ ] Explain standardization vs normalization and when to use each
- [ ] Explain why some algorithms need scaling and others don't
- [ ] Choose between Label Encoding and One-Hot Encoding correctly
- [ ] Create at least 3 engineered features from raw data, with justification
- [ ] Define data leakage and explain how to prevent it in a real pipeline
- [ ] Recite the correct professional data preparation workflow, in order
- [ ] Answer every interview question in this chapter without looking