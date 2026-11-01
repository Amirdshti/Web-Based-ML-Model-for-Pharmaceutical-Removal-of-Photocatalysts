XGBoost-PSO Streamlit App — Five-Fold CV, SHAP, PDP and Excel Exports

Files:
  XGBoost_PSO_Streamlit_5Fold_SHAP_PDP.py
  requirements_XGB_5Fold.txt

Setup:
  pip install -r requirements_XGB_5Fold.txt
  streamlit run XGBoost_PSO_Streamlit_5Fold_SHAP_PDP.py

App workflow:
  1. Download the SampleData.xlsx template in the app, and upload your own completed Excel dataset.
  2. Choose holdout training percentage, PSO or manual XGBoost, and random seed.
  3. In the sidebar, choose five-fold CV on "Training subset only" (default) or "Entire uploaded dataset".
  4. Leave "Generate 5-fold SHAP and PDP" enabled to compute out-of-fold SHAP explanations
     and five fold-specific PDP curves ordered by pooled mean absolute OOF SHAP.
  5. Click Develop/Redevelop, then download Excel results and ZIP figures.

Excel workbook tabs:
  Holdout Predictions
  Holdout Metrics
  Hyperparameters
  CV Fold Metrics
  CV Summary
  CV OOF Predictions
  CV SHAP Importance
  CV SHAP Values
  CV PDP Curves
  CV Methodology

Interpretation:
  Five-fold models share hyperparameters chosen using the original development subset.
  In PSO mode, these are post-tuning diagnostics, not nested cross-validation.
  "Entire uploaded dataset" CV uses holdout rows in this separate experiment;
  do not treat it and the original holdout result as two independent validations.
  SHAP uses validation-row explanations only, and backgrounds sampled from each
  fold's training rows. PDP reference rows come from each fold's training partition.

Data needed:
  The application computes actual metrics and figures only after you upload an
  experimental Excel data file. No real dataset accompanies this package.
