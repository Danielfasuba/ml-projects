## Titanic Survival Prediction (v2)

### Overview
Predict whether a Titanic passenger survived from ticket class, sex, age, family aboard and fare, using Kaggle's labelled training data (891 passengers).

### Why there is a v2: auditing my own model
My first version reported **98.8% test accuracy**. When I audited it, that figure turned out to be an artefact, not a result:

- **Label leakage.** It was trained and tested on `tested.csv`, a copy of Kaggle's *test* set. Its `Survived` column comes from Kaggle's `gender_submission.csv`, a sample answer in which every woman survives and every man does not. The model simply learned *sex = survival*. The tell: 36.4% of the file is marked as survived, exactly the share of women in it (152 of 418).
- **Preprocessing fitted on test data.** `fit_transform` was called on the test set, so the scaler learned from data the model should never see.
- **An ID used as a feature.** `PassengerId` is a row number, not passenger information.

### Approach
- **Held-out test set first:** a stratified 80/20 split made before any exploration (712 train / 179 test).
- **Feature engineering:**
  - **Title:** extracted from the passenger's name.
  - **FamilySize** and **IsAlone:** derived from the siblings/spouses and parents/children columns.
  - **HasCabin:** whether a cabin is recorded.
- **Single scikit-learn pipeline:** median imputation, scaling and one-hot encoding are all fitted inside it, so every cross-validation fold is clean.
- **Baselines:**
  - **Majority class** (everyone died): 61.7%.
  - **"Women survive" rule:** 78.9% on train, 77.7% on test.
- **Model comparison (5-fold stratified CV):** logistic regression, RBF SVM, random forest and gradient boosting.
- **Tuning:** a grid search on the two strongest models.

### Results

| Model | Train accuracy | CV accuracy | CV ROC-AUC |
|---|---|---|---|
| SVM (RBF) | 0.853 | 0.831 | 0.867 |
| Logistic regression | 0.836 | 0.830 | 0.872 |
| Gradient boosting | 0.924 | 0.824 | 0.895 |
| Random forest | 0.986 | 0.823 | 0.876 |

The random forest's 0.986 training accuracy against 0.823 in cross-validation is a clear sign of overfitting. Models were judged on cross-validated scores only.

**Final model:** a tuned SVM (C=3, gamma=0.03), scored once on the untouched test set:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **84.4%** | 82.5% | 75.4% | 78.8% | 0.849 |

That's 6.7 points better than the "women survive" baseline on the same passengers.

### Error analysis
28 of 179 test passengers were misclassified. Most errors are passengers the "women and children first" pattern doesn't explain: men who survived (17) and third-class women who didn't (9).

### Lessons
1. Check where the labels come from.
2. Split off the test set before exploring or preprocessing.
3. Keep every transformation inside a pipeline.
4. Always compare against a simple baseline.
5. Treat an unusually high score as a bug until proven otherwise.

- **Data:** Kaggle *Titanic: Machine Learning from Disaster*, `train.csv`
