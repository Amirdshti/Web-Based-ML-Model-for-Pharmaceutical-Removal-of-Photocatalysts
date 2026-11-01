train_row = metrics_df.loc[metrics_df["Dataset"] == "Train"].iloc[0]
    test_row = metrics_df.loc[metrics_df["Dataset"] == "Test"].iloc[0]
    total_row = metrics_df.loc[metrics_df["Dataset"] == "Total"].iloc[0]

    st.divider()
    st.header("Developed Model Results")
    st.write(
        f"Model: **{st.session_state.model_name}** | "
        f"Attempt: **{st.session_state.attempt}** | "
        f"Random seed used: **{st.session_state.random_seed_used}**"
    )

    if np.isfinite(st.session_state.best_cv_r2):
        st.metric("Best PSO cross-validation R²", f"{st.session_state.best_cv_r2:.4f}")

    st.subheader("Statistical Performance")
    r2_cols = st.columns(3)
    r2_cols[0].metric("Train R²", f"{train_row['R2']:.4f}")
    r2_cols[1].metric("Test R²", f"{test_row['R2']:.4f}")
    r2_cols[2].metric("Total R²", f"{total_row['R2']:.4f}")

    rmse_cols = st.columns(3)
    rmse_cols[0].metric("Train RMSE", f"{train_row['RMSE']:.4f}")
    rmse_cols[1].metric("Test RMSE", f"{test_row['RMSE']:.4f}")
    rmse_cols[2].metric("Total RMSE", f"{total_row['RMSE']:.4f}")

    mae_cols = st.columns(3)
    mae_cols[0].metric("Train MAE", f"{train_row['MAE']:.4f}")
    mae_cols[1].metric("Test MAE", f"{test_row['MAE']:.4f}")
    mae_cols[2].metric("Total MAE", f"{total_row['MAE']:.4f}")

    mape_cols = st.columns(3)
    mape_cols[0].metric("Train MAPE (%)", f"{train_row['MAPE (%)']:.2f}")
    mape_cols[1].metric("Test MAPE (%)", f"{test_row['MAPE (%)']:.2f}")
    mape_cols[2].metric("Total MAPE (%)", f"{total_row['MAPE (%)']:.2f}")

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Experimental versus Predicted Plot")
    st.image(st.session_state.plot_png)

    st.subheader("Optimized Model Hyperparameters")
    st.dataframe(st.session_state.params_df, use_container_width=True, hide_index=True)

    with st.expander("Prediction results", expanded=False):
        st.dataframe(st.session_state.results_df, use_container_width=True)

    st.divider()
    st.subheader("Five-Fold Cross-Validation Results")
    st.caption(f"CV scope: **{cv_scope}**. Predictions are out-of-fold for this selected scope. "
               "PSO-tuned five-fold scores are post-selection diagnostics, not nested CV estimates.")
    st.dataframe(st.session_state.cv_summary_df, hide_index=True, use_container_width=True)
    with st.expander("Fold-by-fold train and validation metrics"):
        st.dataframe(st.session_state.cv_metrics_df, hide_index=True, use_container_width=True)
    with st.expander("Out-of-fold predictions and original Excel row numbers"):
        st.dataframe(st.session_state.cv_oof_df, hide_index=True, use_container_width=True)
    st.image(st.session_state.cv_plots["CV5_OOF_scatter.png"], caption="Five-fold CV out-of-fold predictions")
    if "CV5_SHAP_beeswarm.png" in st.session_state.cv_plots:
        st.subheader("Five-Fold Out-of-Fold SHAP Analysis")
        st.caption("Each record is explained only using the model that did not train on it. "
                   "The SHAP background is drawn from that model's training fold. "
                   "Inputs that are constant in the data are omitted from the figures.")
        st.image(st.session_state.cv_plots["CV5_SHAP_beeswarm.png"])
        st.image(st.session_state.cv_plots["CV5_SHAP_importance.png"])
        st.dataframe(st.session_state.cv_shap_importance_df, hide_index=True,
                     use_container_width=True)
        st.subheader("Five-Fold Partial Dependence Plots")
        st.caption("PDP panels appear from highest to lowest mean absolute held-out SHAP importance. "
                   "Each curve represents one of the five fold-specific models.")
        st.image(st.session_state.cv_plots["CV5_PDP_SHAP_order.png"])
        with st.expander("Numerical PDP data (also in the Excel file)"):
            st.dataframe(st.session_state.cv_pdp_df, hide_index=True, use_container_width=True)
    with st.expander("Interpretation and validation limitations"):
        for line in st.session_state.cv_explanation:
            st.write("• " + line)

    st.subheader("Download the Developed Model and Results")
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download Excel Results",
        data=st.session_state.excel_results,
        file_name="XGBoost_PSO_Holdout_5Fold_SHAP_PDP_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    download_cols[1].download_button(
        "Download Trained Model",
        data=st.session_state.model_file,
        file_name="Trained_XGBoost_PSO_Model.pkl",
        mime="application/octet-stream",
    )
    download_cols[2].download_button(
        "Download R² Plot",
        data=st.session_state.plot_png,
        file_name="XGBoost_PSO_R2_Plot.png",
        mime="image/png",
    )

    st.download_button(
        "Download Five-Fold CV + SHAP + PDP Figures (ZIP)",
        data=st.session_state.cv_plots_zip,
        file_name="XGBoost_PSO_5Fold_SHAP_PDP_Plots.zip",
        mime="application/zip",
    )
    st.caption("The Excel workbook includes CV fold metrics, OOF predictions, "
               "SHAP values and rankings, and all five PDP curves as numerical tables.")

    if not np.isfinite(float(test_row["R2"])):
        st.warning("Test R² could not be calculated for the current test subset.")
    elif float(test_row["R2"]) < 0.80:
        st.warning(
            "The test R² is below 0.80. Review the dataset and PSO settings. Clicking "
            "Develop / Redevelop again uses the next random seed and therefore a different "
            "train/test split; report the five-fold CV summary rather than the best of "
            "several splits."
        )
    else:
        st.info("Test R² is at least 0.80 for this split. Because a single split can be "
                "favourable, report the five-fold CV summary alongside it, and consider "
                "external study-group validation before claiming generalisation.")