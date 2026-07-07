### **Alternate Credit Scoring System — Exact Pipeline Specification**

---

#### **Preliminary — Data Split (Done Before Any Stage)**

The very first operation after loading the CSV — before feature engineering, before anything — is the train/test split. This order is non-negotiable.

Load alternate\_credit\_raw.csv  
→ Separate X (all columns except 'defaulted' and 'default\_probability') and y ('defaulted')  
→ Retain 'default\_probability' as a separate series for calibration analysis later  
→ stratified train/test split: 80% train, 20% test, random\_state=42, stratify=y  
→ Result: X\_train (8,000 rows), X\_test (2,000 rows), y\_train, y\_test  
→ Also retain borrower\_type column separately from X\_train and X\_test before engineering  
   (needed for group-wise imputation and fairness audit — store as borrower\_type\_train, borrower\_type\_test)

All fitting operations — imputers, scalers, encoders, clustering, SMOTE — are fit exclusively on training data. Test data is only ever transformed, never fit on.

---

#### **Stage 1 — Feature Engineering**

**Input:** X\_train, X\_test (raw, with NaNs), borrower\_type\_train, borrower\_type\_test

---

##### **Step 1 — Compute digital\_footprint\_density (Pre-Imputation)**

This must be computed before imputation because it measures how many digital fields are actually present per borrower. After imputation, missingness is gone and the signal is lost.

Digital fields to check: `upi_transactions_per_month`, `upi_avg_transaction_amount`, `upi_months_active`, `mobile_wallet_used`, `ecomm_orders_per_month`, `prepaid_orders_ratio`

digital\_footprint\_density \= (count of non-NaN values across these 6 fields) / 6

Compute this on X\_train and X\_test separately. Store as a new column in both.

---

##### **Step 2 — Create Missingness Indicator Flags (Pre-Imputation)**

Create the following binary columns before any imputation. 1 \= data was missing, 0 \= data was present. Do this on both X\_train and X\_test.

* `upi_data_missing` — 1 if `upi_transactions_per_month` is NaN  
* `rent_data_missing` — 1 if `total_rental_months` is NaN  
* `ecomm_data_missing` — 1 if `ecomm_orders_per_month` is NaN or 0

These three columns become permanent model features.

---

##### **Step 3 — Imputation**

Two tiers. All imputers are fit on X\_train only, then applied to both X\_train and X\_test.

**Tier 1 — Simple median imputation** for fields with low, approximately random missingness:

* `same_number_since_year`  
* `survey_q4`  
* `survey_q6`

Fit `sklearn.impute.SimpleImputer(strategy='median')` on X\_train for these three columns. Transform both X\_train and X\_test.

**Tier 2 — Group-wise imputation by borrower\_type** for fields with structured, segment-driven missingness:

* `upi_transactions_per_month`  
* `upi_avg_transaction_amount`  
* `upi_months_active`  
* `mobile_wallet_used`  
* `rent_paid_on_time_months`  
* `total_rental_months`

For each of these fields, compute the median separately for gig, rural, and migrant borrowers using X\_train \+ borrower\_type\_train only. Then fill NaNs in X\_train and X\_test using the corresponding borrower\_type median. Do not use sklearn for this — implement manually using groupby on the training set.

**Ecomm special case:** `ecomm_return_rate` and `prepaid_orders_ratio` are NaN when `ecomm_orders_per_month` is 0 or NaN. Impute these with 0.0 — a borrower with no e-commerce activity has a return rate and prepaid ratio of 0 by definition.

After imputation, assert that no NaN values remain in X\_train or X\_test. If any do, raise an error and investigate.

---

##### **Step 4 — Derived Feature Computation**

Compute all of the following on both X\_train and X\_test after imputation. These replace or augment raw fields — keep the raw fields unless specified otherwise.

**Payment features:**

* `utility_payment_ratio` \= `utility_bills_paid` / `utility_bills_total`. If `utility_bills_total` \= 0, set to 0.5.  
* `rent_consistency_score` \= `rent_paid_on_time_months` / `total_rental_months`. If `rent_data_missing` \= 1 (meaning rent fields were genuinely absent), set this to 0.5 and rely on the `rent_data_missing` flag to carry the signal.

**Income features:**

* `income_mean` \= mean of `income_month_1` through `income_month_6`  
* `income_std` \= std of `income_month_1` through `income_month_6`  
* `income_regularity_index` \= 1 − (`income_std` / `income_mean`), clipped to \[0, 1\]. If `income_mean` \= 0, set to 0\.  
* `income_trend` \= `income_month_6` − `income_month_1` — captures whether income is growing or declining. Positive is good for migrants especially.

After computing these, drop `income_month_1` through `income_month_6` from the feature matrix — they've served their purpose and their presence alongside the derived features creates redundancy.

**Telecom feature:**

* `telecom_stability_years` \= 2024 − `same_number_since_year`. After computing, drop `same_number_since_year`.

**Loan feature:**

* `loan_to_income_ratio` \= `loan_amount_requested` / (`income_mean` × `loan_tenure_months`). If denominator \= 0, set to 999 (extreme stress signal).

**UPI feature:**

* `upi_consistency_score` \= (`upi_transactions_per_month` / 120\) × (`upi_months_active` / 36). Both components are already imputed. Result is in \[0, 1\]. After computing, keep this and drop `upi_transactions_per_month` and `upi_months_active` — they are now encoded in the composite.

---

##### **Step 5 — Psychometric Composite Scores**

Compute on both X\_train and X\_test. Q3 and Q7 are reverse-scored — use (6 − answer) before including in any average.

* `financial_discipline_score` \= mean(Q1, Q2, (6−Q3), Q4, Q8) — 5 questions, equal weight  
* `future_planning_score` \= mean(Q5, Q6) — 2 questions  
* `risk_appetite_score` \= mean((6−Q3), (6−Q7)) — 2 reverse-scored questions. Higher \= more risk-averse \= positive creditworthiness signal.

Note Q3 contributes to both `financial_discipline_score` and `risk_appetite_score` — this is intentional, it's a strong signal. After computing composites, drop `survey_q1` through `survey_q8` from the feature matrix — they've been encoded into the three composites.

---

##### **Step 6 — Encoding**

**State → Region mapping first.** Map the 10–12 states into 5 regions before encoding:

* North: Delhi, Uttar Pradesh, Rajasthan  
* South: Karnataka, Tamil Nadu, Telangana  
* West: Maharashtra, Gujarat  
* East: Bihar, Odisha  
* Central: Madhya Pradesh

Then one-hot encode: `region`, `borrower_type`, `employment_type`, `loan_purpose`. Use `sklearn.preprocessing.OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`. Fit on X\_train only, transform both. Drop the original categorical columns after encoding.

Do not one-hot encode the binned columns yet — those come next.

---

##### **Step 7 — Discretization**

Compute bin boundaries using X\_train only. Apply same boundaries to X\_test.

**loan\_amount\_requested → loan\_amount\_bin:** Use `pd.qcut` on X\_train's `loan_amount_requested` with q=4 (quartile-based). Labels: 'Small', 'Medium', 'Large', 'Very\_Large'. Then ordinal encode: Small=0, Medium=1, Large=2, Very\_Large=3. Keep both `loan_amount_requested` (raw) and `loan_amount_bin` (encoded) in the feature matrix.

**income\_mean → income\_level\_bin:** Use `pd.qcut` on X\_train's `income_mean` with q=3. Labels: 'Low', 'Medium', 'High'. Ordinal encode: Low=0, Medium=1, High=2. Keep both.

---

##### **Step 8 — Normalization**

Apply `sklearn.preprocessing.MinMaxScaler()` to all continuous numeric columns. Fit on X\_train only, transform both X\_train and X\_test.

Columns to scale: all continuous features after encoding and derivation. Specifically exclude: all one-hot encoded binary columns (already 0/1), all ordinal bin columns (already ordinal integers), all missingness flag columns (binary).

After scaling, confirm all continuous columns have values in \[0, 1\].

---

##### **Step 9 — PCA (Exploratory Only)**

Fit `sklearn.decomposition.PCA()` on X\_train after scaling. Plot the scree plot — cumulative explained variance ratio vs number of components. Identify how many components explain 85% of variance. Report this number. This is for demonstration and analysis only — do not reduce the feature matrix for the main pipeline. If you want, train one additional Logistic Regression variant on the PCA-reduced matrix and compare its ROC-AUC to the full-feature version as a bonus analysis.

---

**Output of Stage 1:**

* `X_train_eng` — fully engineered, encoded, scaled training feature matrix, no NaNs  
* `X_test_eng` — same transformations applied, no NaNs  
* `y_train`, `y_test` — unchanged binary targets  
* `borrower_type_train`, `borrower_type_test` — retained separately for Stage 5  
* `default_probability_test` — retained separately for Stage 4 calibration

---

#### **Stage 2 — Unsupervised Clustering**

**Input:** `X_train_eng`, `X_test_eng`

---

##### **Step 1 — Select Clustering Features**

From `X_train_eng`, select only these behavioral composite columns for clustering. Do not use any other features:

* `utility_payment_ratio`  
* `rent_consistency_score`  
* `income_regularity_index`  
* `upi_consistency_score`  
* `digital_footprint_density`  
* `financial_discipline_score`  
* `future_planning_score`  
* `risk_appetite_score`

Store as `X_train_clust` and `X_test_clust`. These are already scaled from Stage 1\.

---

##### **Step 2 — Determine Optimal K**

Run K-Means (`sklearn.cluster.KMeans(n_init=10, random_state=42)`) for K \= 2 through 10 on `X_train_clust`. For each K record: inertia and silhouette score (`sklearn.metrics.silhouette_score`). Plot both — elbow curve (inertia vs K) and silhouette score vs K. Select optimal K as the value where silhouette score is highest. In case of a tie, prefer the lower K for interpretability.

---

##### **Step 3 — Fit Final K-Means**

Fit `KMeans(n_clusters=optimal_K, n_init=10, random_state=42)` on `X_train_clust`. Assign cluster labels to X\_train rows. Store the fitted object as `kmeans_model`.

---

##### **Step 4 — Fit GMM and Compare**

Fit `sklearn.mixture.GaussianMixture(n_components=optimal_K, random_state=42)` on `X_train_clust`. Get hard cluster assignments via `.predict()`. Compute silhouette score for GMM assignments. Compare to K-Means silhouette score. Report both. Use whichever has the higher silhouette score as the final cluster assignment. Store the winning model as `cluster_model`.

---

##### **Step 5 — Cluster Profiling**

Using training data only, compute per-cluster means of all 8 clustering features plus the default rate (`y_train`). Present as a table with clusters as rows and features \+ default rate as columns. Each cluster should have a meaningfully distinct profile — name them descriptively (e.g. "Digitally Active Disciplined", "Cash-Based Reliable", "High-Risk Sparse Data").

---

##### **Step 6 — Cross-Borrower-Type Check**

For each cluster, compute the proportion of gig / rural / migrant borrowers. If any cluster is \>75% one borrower type, flag it explicitly in the notebook with a markdown warning. Do not modify the clustering — just document it.

---

##### **Step 7 — Assign Cluster Labels**

Use `cluster_model.predict(X_train_clust)` → `cluster_train_labels` Use `cluster_model.predict(X_test_clust)` → `cluster_test_labels`

Add `cluster_id` column to `X_train_eng` and `X_test_eng`. Treat `cluster_id` as an ordinal integer feature — do not one-hot encode it, since the cluster ordering has implicit meaning from the profiling step.

---

**Output of Stage 2:**

* `X_train_eng` and `X_test_eng` with `cluster_id` column added  
* `cluster_model` (fitted, for use in Stage 6 per-applicant scoring)  
* Elbow plot, silhouette plot, cluster profile table, cross-type distribution table

---

#### **Stage 3 — Model Training**

**Input:** `X_train_eng`, `X_test_eng`, `y_train`, `y_test`

---

##### **SMOTE — Exact Placement**

This is the most critical procedural instruction in the entire pipeline. Read carefully.

SMOTE is applied in **two distinct contexts** and the procedure differs for each:

**Context A — Cross-validation (for hyperparameter tuning and CV score reporting):** SMOTE must be wrapped inside an `imblearn.pipeline.Pipeline` so it is applied inside each fold, never leaking into the validation fold. Use `imblearn.pipeline.Pipeline` (not sklearn's Pipeline — they are different). Structure every model's CV pipeline as:

ImbPipeline(\[  
    ('smote', SMOTE(random\_state=42)),  
    ('classifier', YourClassifier(\*\*params))  
\])

Pass `X_train_eng` and `y_train` (original, unbalanced) into `cross_validate`. Never pass SMOTE-resampled data into `cross_validate`. SMOTE fires fresh inside each fold on that fold's training portion only.

**Context B — Final model fitting (after CV, for the model that gets evaluated on test set):** After CV is complete and best hyperparameters are selected, apply SMOTE once to the full `X_train_eng` / `y_train`:

smote \= SMOTE(random\_state=42)  
X\_train\_sm, y\_train\_sm \= smote.fit\_resample(X\_train\_eng, y\_train)  
final\_model.fit(X\_train\_sm, y\_train\_sm)

Then evaluate on `X_test_eng` / `y_test` — the test set is never touched by SMOTE under any circumstances.

**CV strategy for all models:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Stratified because of class imbalance. Same fold object reused for all models so results are comparable.

**Scoring metrics to collect during CV:** `['accuracy', 'precision', 'recall', 'f1', 'roc_auc']` via `cross_validate`'s `scoring` parameter.

---

##### **Model 1 — Logistic Regression L2**

Hyperparameter tuning: grid search over `C` in `[0.001, 0.01, 0.1, 1, 10, 100]` using the ImbPipeline \+ StratifiedKFold. Select C with best CV ROC-AUC.

Final model: `LogisticRegression(C=best_C, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)`. Fit on `X_train_sm, y_train_sm`.

Extra output: Extract and plot the top 15 feature coefficients by absolute magnitude. These are directly interpretable as feature importance for logistic regression.

---

##### **Model 2 — Logistic Regression L1**

Same grid search over C. Use `solver='liblinear'` (required for L1). Select best C via CV ROC-AUC.

Final model: `LogisticRegression(C=best_C, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)`. Fit on `X_train_sm, y_train_sm`.

Extra output: Count and report how many features have zero coefficients (L1 sparsity). Compare to L2 — this demonstrates the difference between Ridge and Lasso regularization from Unit IV.

---

##### **Model 3 — Naive Bayes**

No hyperparameter tuning needed. `GaussianNB()` directly in the ImbPipeline for CV. Final model fit on `X_train_sm, y_train_sm`. Report that independence assumption is violated here and discuss expected impact on performance.

---

##### **Model 4 — K-Nearest Neighbours**

Hyperparameter tuning: grid search over `n_neighbors` in `[3, 5, 7, 9, 11, 15, 21]`. Select best K via CV ROC-AUC.

Final model: `KNeighborsClassifier(n_neighbors=best_K, metric='minkowski')`. Fit on `X_train_sm, y_train_sm`. Note: KNN is sensitive to scale — confirm normalization from Stage 1 is applied. Discuss curse of dimensionality in the notebook.

---

##### **Model 5 — Decision Tree**

Train two variants:

**Variant A (overfit — for demonstration):** `DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)`. No CV tuning — deliberately overfit. Fit directly on `X_train_sm, y_train_sm`. Report train accuracy vs test accuracy to show overfitting explicitly.

**Variant B (tuned):** Grid search over `max_depth` in `[3, 5, 7, 10, 15]` and `criterion` in `['gini', 'entropy']`. Select best combo via CV ROC-AUC. Final model fit on `X_train_sm, y_train_sm`. All Stage 4 evaluation uses Variant B only. Variant A is reported separately as the overfitting demonstration.

---

##### **Model 6 — Random Forest**

Hyperparameter tuning: grid search over:

* `n_estimators`: \[100, 200, 300\]  
* `max_depth`: \[10, 20, None\]  
* `min_samples_split`: \[2, 5\]

Select best combination via CV ROC-AUC. Final model: `RandomForestClassifier(best_params, random_state=42)`. Fit on `X_train_sm, y_train_sm`.

---

##### **Model 7 — XGBoost**

Hyperparameter tuning: grid search over:

* `n_estimators`: \[100, 200, 300\]  
* `max_depth`: \[3, 5, 7\]  
* `learning_rate`: \[0.01, 0.1, 0.2\]  
* `subsample`: \[0.8, 1.0\]

Select best combination via CV ROC-AUC. Final model: `XGBClassifier(best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)`. Fit on `X_train_sm, y_train_sm`.

---

##### **Model 8 — SVM**

Train on a stratified subsample of 3,000 rows from `X_train_sm, y_train_sm` (SVM does not scale to 10,000+ rows). Use `StratifiedShuffleSplit` to sample. Grid search over `C` in `[0.1, 1, 10]` and `kernel` in `['linear', 'rbf']`. Final model: `SVC(C=best_C, kernel=best_kernel, probability=True, random_state=42)`. The `probability=True` flag is mandatory — needed for ROC-AUC and calibration computation.

---

**Output of Stage 3:**

* 8 fitted final model objects (LR-L2, LR-L1, NB, KNN, DT-overfit, DT-tuned, RF, XGB, SVM)  
* CV score table (mean ± std across 5 folds for each metric, per model)  
* Best hyperparameters per model documented

---

#### **Stage 4 — Model Evaluation**

**Input:** All fitted models, `X_test_eng`, `y_test`, `default_probability_test`, `borrower_type_test`

Evaluate every model (except DT-overfit, which is evaluated separately) on `X_test_eng`. Never refit or modify models here.

---

##### **Step 1 — Per-Model Metrics**

For each model compute:

* `y_pred` \= `.predict(X_test_eng)` — hard class predictions at default threshold 0.5  
* `y_prob` \= `.predict_proba(X_test_eng)[:, 1]` — predicted default probabilities

From these compute: accuracy, precision, recall, F1, ROC-AUC, average precision (area under PR curve), Brier score. Confusion matrix with raw counts (not rates).

---

##### **Step 2 — Threshold Optimisation (Best Model Only)**

For the best model by ROC-AUC, do not rely on the default 0.5 threshold. Instead:

Compute precision and recall at every threshold from 0.0 to 1.0 in steps of 0.01 using `precision_recall_curve`. Find the threshold that maximises F1. Also find the threshold that achieves recall ≥ 0.80 with the highest precision at that constraint (the loan officer's conservative threshold). Report both thresholds. Re-compute the confusion matrix and all metrics at each threshold. Report all three threshold variants (0.5, max-F1, recall≥0.80) in a table side by side.

---

##### **Step 3 — Plots**

Generate these plots:

* ROC curves for all models on one plot, with AUC in the legend  
* Precision-Recall curves for all models on one plot  
* Confusion matrix heatmap for each model individually (raw counts)  
* Reliability diagram (calibration plot) for the best model — use `sklearn.calibration.calibration_curve` with `n_bins=10`. Plot predicted probability vs fraction of positives. If poorly calibrated, apply `sklearn.calibration.CalibratedClassifierCV` with `method='isotonic'` and report whether calibration improved the Brier score.

---

##### **Step 4 — Comparative Summary Table**

One table, models as rows, all metrics as columns: Test Accuracy, Test Precision, Test Recall, Test F1, Test ROC-AUC, Test Avg Precision, Brier Score, CV ROC-AUC (mean), CV ROC-AUC (std). Highlight the best value in each column. This is the centrepiece of Stage 4\.

---

##### **Step 5 — Segment-wise Evaluation**

For the best model only, split `X_test_eng` by `borrower_type_test` (gig / rural / migrant). Compute precision, recall, F1, and ROC-AUC separately for each segment. Report as a table. Flag any segment where recall is below 0.60 — this indicates the model is missing a disproportionate number of actual defaulters from that group.

---

##### **Step 6 — Select Final Best Model**

Make the final model selection explicitly with written justification based on the test metrics. Primary criterion: ROC-AUC. Tiebreaker: average precision (PR-AUC). Secondary consideration: Brier score (calibration quality). Name this model `best_model` — this object is used in Stages 5 and 6\.

---

**Output of Stage 4:**

* All plots  
* Comparative summary table  
* Threshold analysis table for best model  
* Segment-wise evaluation table  
* `best_model` object named and justified

---

#### **Stage 5 — Fairness & Bias Audit**

**Input:** `best_model`, `X_test_eng`, `y_test`, `borrower_type_test`

Get predictions: `y_pred_best` \= `best_model.predict(X_test_eng)` at the max-F1 threshold from Stage 4 (not 0.5).

Compute the following for each borrower type (gig, rural, migrant):

* **Predicted positive rate** \= proportion of borrowers predicted as defaulters. Demographic parity requires this to be similar across groups.  
* **True Positive Rate (Recall)** \= TP / (TP \+ FN). Equalized odds requires similar TPR across groups.  
* **False Positive Rate** \= FP / (FP \+ TN). Equalized odds also requires similar FPR across groups.  
* **Precision** per group.

Present all four metrics per group in one table. For each metric, compute the ratio of the highest group value to the lowest group value — a ratio above 1.25 is worth flagging as a meaningful disparity.

Write a markdown interpretation cell discussing: whether disparities reflect the data (rural borrowers genuinely have higher default rates in the synthetic data) versus model bias (the model is over-penalising a group beyond what features justify). Do not attempt to fix the bias — analyse and document it.

---

**Output of Stage 5:**

* Fairness metrics table with disparity ratios  
* Written interpretation

---

#### **Stage 6 — Explainability & Output Report**

**Input:** `best_model`, `X_train_eng`, `X_test_eng`, `y_test`, `cluster_model`, column names

---

##### **Step 1 — SHAP Values**

Use `shap.TreeExplainer(best_model)` if best model is tree-based (RF or XGB). If best model is Logistic Regression, use `shap.LinearExplainer`. If SVM, use `shap.KernelExplainer` on a 200-row subsample (KernelExplainer is slow).

Compute SHAP values on the full `X_test_eng`. Store as `shap_values` array of shape (n\_test, n\_features).

---

##### **Step 2 — Global SHAP Plots**

* **Beeswarm summary plot:** `shap.summary_plot(shap_values, X_test_eng)` — shows feature importance and direction globally  
* **Bar summary plot:** `shap.summary_plot(shap_values, X_test_eng, plot_type='bar')` — mean absolute SHAP values per feature, ranked  
* **Dependence plots:** For the top 3 features by mean absolute SHAP value, generate `shap.dependence_plot(feature, shap_values, X_test_eng)` — shows how that feature's value relates to its SHAP contribution

---

##### **Step 3 — Per-Applicant Report Function**

Write a function `generate_applicant_report(applicant_index, X_test_eng, shap_values, y_test, cluster_model, cluster_profiles)` that returns a dictionary with exactly these keys:

* `borrower_id` — from the original dataset index  
* `default_probability` — `best_model.predict_proba(applicant_row)[:, 1][0]`, rounded to 4 decimal places  
* `risk_tier` — 'Low' if probability \< 0.30, 'Medium' if 0.30–0.60, 'High' if \> 0.60  
* `actual_outcome` — `y_test.iloc[applicant_index]` (for validation during testing)  
* `top_3_positive_factors` — top 3 features with the most negative SHAP values (reducing default probability), returned as list of plain-English strings mapped from feature names  
* `top_3_negative_factors` — top 3 features with the most positive SHAP values (increasing default probability), same format  
* `data_confidence_flag` — computed from the applicant's `digital_footprint_density` and missingness flags: High if density ≥ 0.8 and both `upi_data_missing` \= 0 and `ecomm_data_missing` \= 0; Low if density \< 0.4 or two or more missingness flags \= 1; Medium otherwise  
* `cluster_id` — the applicant's cluster assignment  
* `cluster_profile_name` — the descriptive name assigned to that cluster in Stage 2

Create a feature name → plain English mapping dictionary explicitly. For example: `'utility_payment_ratio': 'Utility bill payment consistency'`, `'income_regularity_index': 'Income stability over 6 months'`, etc. Cover all features.

---

##### **Step 4 — Waterfall Plot**

For a single applicant (demonstrate on one defaulter and one non-defaulter from the test set), generate `shap.waterfall_plot(shap.Explanation(values=shap_values[i], base_values=explainer.expected_value, data=X_test_eng.iloc[i], feature_names=feature_names))`.

---

##### **Step 5 — Natural Language Summary**

Write a `generate_nl_summary(report_dict)` function that takes the report dictionary and returns a paragraph string using a template with conditional logic. The template must:

* State borrower type and risk tier in the opening sentence  
* Name the top 2 positive factors in plain English  
* Name the top 2 negative factors in plain English  
* State the data confidence flag and what it means  
* Close with a recommendation: "Recommended for approval" if Low risk, "Recommended for manual review" if Medium risk, "Recommended for rejection" if High risk

Demonstrate this function on the same two applicants used for the waterfall plot.

---

**Output of Stage 6:**

* Global SHAP plots (beeswarm, bar, dependence)  
* Waterfall plots for two applicants  
* `generate_applicant_report()` function  
* `generate_nl_summary()` function  
* Two fully printed example reports with NL summaries

