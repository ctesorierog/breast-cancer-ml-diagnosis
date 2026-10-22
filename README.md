# Breast cancer diagnosis: from hidden structure to prediction

Supervised and unsupervised machine learning on the **Breast Cancer Wisconsin (Diagnostic)** dataset: 569 tumours, 30 cell-nucleus features each. The project first asks whether the benign/malignant split shows up without labels (PCA, t-SNE, clustering), then builds classifiers and links the two parts through error analysis.

## Results

**Final model: SVM (RBF kernel) with PCA inside the pipeline.** On the untouched test set (114 tumours, malignant = positive class):

| Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| 0.982 | 1.000 | 0.952 | 0.976 | 0.997 |

All 72 benign tumours were classified correctly. The two errors were both false negatives, and both lie in the region where the clusters overlap.

## What the notebook does

1. **Data checks and EDA**: no missing values or duplicates, class balance of 63 % benign and 37 % malignant, very different feature scales, strong correlation between the size-related features.
2. **Leakage-safe setup**: a stratified 80/20 split. Scaling and PCA are fitted inside the pipelines only.
3. **Unsupervised learning**
   - PCA: PC1 and PC2 explain 63 % of the variance. 7 components keep 90 % and 10 keep 95 %.
   - t-SNE with several perplexities.
   - K-Means (ARI 0.66, silhouette 0.34) and a Gaussian Mixture Model (ARI 0.75, silhouette 0.31). Both find the diagnosis without being given labels.
4. **Supervised learning** with stratified 5-fold CV, tuned on F1 for the malignant class:

   | Model | Original features | With PCA |
   |---|---|---|
   | Logistic Regression | 0.964 | 0.967 (10 components) |
   | SVM RBF | 0.967 | **0.970** (15 components) |
   | kNN | 0.963 | 0.964 (6 components) |

5. **Connecting both parts**: adding GMM cluster probabilities as features did not improve the model. Mapping the final model's errors onto PCA and GMM space shows that they sit at the boundary between clusters.

## Files

- `breast_cancer_ml.ipynb`: full analysis with outputs and discussion.
- `docs/report.pdf`: final report.

## Stack

Python, scikit-learn (PCA, t-SNE, K-Means, GMM, Logistic Regression, SVM, kNN, GridSearchCV), pandas, matplotlib, seaborn, Google Colab. The dataset ships with scikit-learn (`load_breast_cancer`).

## Team

Carlos Tesoriero and Felipe Salazar Molina. Final project for *Machine Learning*, AI Engineering, Tecnun – Universidad de Navarra, May 2026.
