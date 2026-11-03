# EECE 490/690 – Ensemble Learning, Member 2 (Parts 3 & 4)

## Context
Group assignment (AUB, Fall 2026–27). Dataset B: Heart Disease (918 rows, `heart.csv`).
Three members; I am **Member 2: modeling**. My deliverable is the section of the shared
notebook covering **Part 3 (Random Forest vs. boosting)** and **Part 4 (class imbalance with SMOTENC)**.
Member 3 later uses my results for Part 5 (evaluation), Part 6 (SHAP) and Part 7 (conclusions).

Full specs: `docs/Assignment.txt` (official) and `docs/Group_Plan.txt` (our split of work).
Member 1's finished section (EDA + preprocessing, with outputs): `reference/Member1_Setup_EDA_Preprocessing.ipynb`.

## Working file
`Member2_Models_SMOTENC.ipynb`: a skeleton with headings, requirements and TODO cells.
Fill the TODO cells; keep the structure. Final notebook must run top to bottom (Restart & Run All)
and keep all outputs, because it will be pasted into the group's master notebook and run in Google Colab.

## Hard rules (graded)
- Import everything shared from `shared_setup.py` (`from shared_setup import *`). **Do not edit `shared_setup.py`**: it is Member 1's file and the whole team depends on it.
- Use `load_and_split("heart.csv")` for data and the shared `CV` object for every CV call → identical folds for all models.
- **Never touch `X_test` / `y_test`.** They are reserved for Member 3's single final evaluation in Part 5.
- All preprocessing lives inside a pipeline so it is refit per training fold (no leakage). Use `build_preprocessor("improved")` (Member 1's choice) unless an experiment explicitly compares otherwise.
- Always `return_train_score=True`; summarise with `summarize_cv()` so tables match Member 1's format (`train_*` / `val_*`, mean ± std).
- Fixed seed `SEED = 42` for every model and sampler.
- Change **one factor at a time** in every comparison.

## Part 3 requirements
- Models: Random Forest, AdaBoost (defaults allowed), XGBoost; keep the DummyClassifier as reference.
- Tune **two** hyperparameters for RF and **two** for XGBoost, **max 6 candidate settings each** (plan: RF `max_depth` {None,5,10} × `min_samples_leaf` {1,5}; XGB `max_depth` {3,5} × `learning_rate` {0.05,0.1,0.3}).
- Justify the metric used to select the best setting (primary metric of the project is recall on class 1, ROC-AUC secondary; note that recall alone is maximised trivially by the dummy).
- At least one visualisation; interpretation markdown after each experiment.
- Answer the guiding questions: why can boosting outperform bagging? when can boosting overfit? did extra complexity help?

## Part 4 requirements
- Report the class ratio (which class is the minority?).
- One controlled experiment: same model with vs. without **SMOTENC**, using an `imblearn.pipeline.Pipeline`.
- Correct ordering: impute + scale numerics (reuse `build_numeric_pipeline("improved")`), keep categoricals **un-encoded** for SMOTENC, resample, then one-hot encode, then the model. Never run SMOTE on one-hot columns; SMOTENC cannot take NaN.
- Resampling only inside training folds; never on validation or test data.
- Discuss whether resampling was needed (no improvement is a valid result) and the guiding question: can recall improve while precision/accuracy worsens?

## Handoff to Member 3 (end of notebook)
State the recommended final-model candidate (from validation only), expose the fitted search objects / tables, and save all CV results to `member2_cv_results.csv`.

## Environment
`pip install -r requirements.txt` (scikit-learn, imbalanced-learn, xgboost, pandas, numpy, matplotlib, seaborn, jupyter).
Run `python check_env.py` first: it verifies packages, the shared module and the data split.
Execute the notebook headless with:
`jupyter nbconvert --to notebook --execute --inplace Member2_Models_SMOTENC.ipynb`

## How I want help
I want to understand and do this work myself. Explain the reasoning behind each step (not just the code),
and check my results against the rules above. Every number in the markdown must come from an actual executed output.

## Step protocol (run after every step)
1. Restart and run the whole notebook top to bottom (jupyter nbconvert --to notebook --execute --inplace).
2. Report: (a) any errors; (b) whether outputs of cells from earlier steps changed (they must not); (c) whether X_test or y_test is used anywhere (must be no); (d) a table matching every number in new markdown to the output it came from; (e) the step's expected values vs actual values.
3. If anything fails or an expected value differs, STOP: do not continue, do not fix silently. Explain what is wrong.
4. Never commit or push on your own. When I reply "PASS", commit with message "checkpoint: <step name>" and push. When I reply "FAIL", restore the notebook from the last commit with git and wait for instructions.
