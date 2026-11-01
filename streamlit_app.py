# -*- coding: utf-8 -*-
"""
Streamlit application for developing XGBoost-PSO models from a spreadsheet
with 11 inputs and 1 output, including holdout plus post-selection five-fold CV,
out-of-fold SHAP, fold-model PDP and downloadable Excel/figure outputs.

Workflow
1. Download the Excel template.
2. Fill and upload the completed dataset.
3. Select the train/test ratio and model settings.
4. Develop or redevelop an XGBoost model, with optional PSO optimization.
   Each Develop/Redevelop click uses the next random seed, i.e. a new split.
5. Display R², RMSE, MAE and MAPE for train, test and total data.
6. Display an experimental-versus-predicted plot and optimized hyperparameters.
7. Assess five-fold cross-validation (folds different from the PSO search folds)
   and out-of-fold SHAP/PDP.
8. Download Excel tables, SHAP/PDP figures, and trained model.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="XGBoost-PSO Pharmaceutical Photodegradation Model",
    layout="wide",
)

st.title("Web-Based XGBoost–PSO Model Development App")
st.write(
    "Download the Excel template, enter the complete modelling dataset, upload it, "
    "select the train/test ratio, and develop or redevelop an XGBoost–PSO model for "
    "pharmaceutical degradation or removal by photocatalysts."
)


# ============================================================
# Exact data structure: 11 inputs + 1 output
# ============================================================
INPUT_COLUMNS = [
    "BET specific surface area (m2 g-1)",
    "Oxidant concentration (mM)",
    "Molecular Weight (g/mol)",
    "HBDC",
    "HBAC",
    "Topological Polar Surface Area (\u212b\u00b2)",  # same Å character as SampleData.xlsx
    "Initial concentration of pollutant (mg/L)",
    "pH",
    "Light source",
    "Catalyst dosage (mg/L)",
    "t (min)",
]

OUTPUT_COLUMN = "Degradation or Removal (%)"
TEMPLATE_COLUMNS = INPUT_COLUMNS + [OUTPUT_COLUMN]

# Alternative header spellings accepted on upload (matched after normalisation).
COLUMN_ALIASES = {
    "Solution pH": "pH",
}

LIGHT_SOURCE_MAP = {
    "uv": 1.0,
    "ultraviolet": 1.0,
    "ultraviolet light": 1.0,
    "visible": 2.0,
    "visible light": 2.0,
    "vis": 2.0,
    "simulated solar": 3.0,
    "simulated solar light": 3.0,
    "solar": 3.0,
    "solar light": 3.0,
}


# ============================================================
# Session state
# ============================================================
DEFAULT_STATE: dict[str, Any] = {
    "attempt": 0,
    "model_trained": False,
    "dataset_signature": None,
    "settings_signature": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_model_results() -> None:
    """Remove results when the uploaded dataset or the settings change."""
    keys_to_remove = [
        "model",
        "results_df",
        "metrics_df",
        "params_df",
        "y_train",
        "y_test",
        "y_train_pred",
        "y_test_pred",
        "plot_png",
        "excel_results",
        "model_file",
        "best_cv_r2",
        "model_name",
        "random_seed_used",
        "cv_metrics_df", "cv_summary_df", "cv_oof_df",
        "cv_shap_importance_df", "cv_shap_values_df", "cv_pdp_df",
        "cv_plots", "cv_plots_zip", "cv_explanation",
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)
    st.session_state.model_trained = False
    st.session_state.attempt = 0


# ============================================================
# Utility functions
# ============================================================
def normalise_header(name: Any) -> str:
    """Case-, whitespace- and Unicode-insensitive key for matching headers.

    NFC normalisation folds look-alike characters such as the angstrom sign
    (U+212B) into the standard Å (U+00C5), so headers that look identical are
    also matched as identical. Extra spaces and letter case are ignored.
    """
    text = unicodedata.normalize("NFC", str(name))
    return " ".join(text.split()).casefold()


def standardise_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rename uploaded headers to the exact template names.

    Returns the renamed frame and any template variables that now appear in
    more than one column (for example both 'pH' and 'Solution pH').
    """
    lookup = {normalise_header(c): c for c in TEMPLATE_COLUMNS}
    lookup.update({normalise_header(a): target for a, target in COLUMN_ALIASES.items()})
    renamed = frame.rename(
        columns={c: lookup.get(normalise_header(c), c) for c in frame.columns}
    )
    duplicated = sorted({
        str(c) for c in renamed.columns[renamed.columns.duplicated()]
        if c in TEMPLATE_COLUMNS
    })
    return renamed, duplicated


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in %, excluding observations with a zero experimental value."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    nonzero = np.abs(y_true) > np.finfo(float).eps
    if not np.any(nonzero):
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
) -> dict[str, Any]:
    return {
        "Dataset": dataset_name,
        "Number of data points": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": compute_mape(y_true, y_pred),
    }


def encode_light_source(series: pd.Series) -> pd.Series:
    """Accept numeric codes or common light-source text labels."""
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower().map(LIGHT_SOURCE_MAP)
    encoded = numeric.fillna(text)
    return encoded


def create_fallback_template() -> bytes:
    """Generate a template only when SampleData.xlsx is absent."""
    output = io.BytesIO()
    template_df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    instructions_df = pd.DataFrame(
        {
            "Item": ["Light-source coding", "Output", "Required structure"],
            "Instruction": [
                "Use 1 = UV, 2 = visible light, or 3 = simulated solar light.",
                "Enter the experimental degradation or removal percentage.",
                "Keep the 11 input columns and the final output column unchanged.",
            ],
        }
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, sheet_name="Data Template", index=False)
        instructions_df.to_excel(writer, sheet_name="Instructions", index=False)
    output.seek(0)
    return output.getvalue()


def get_template_bytes() -> bytes:
    template_path = Path(__file__).with_name("SampleData.xlsx")
    if template_path.exists():
        return template_path.read_bytes()
    return create_fallback_template()


def params_from_position(position: np.ndarray) -> dict[str, Any]:
    """Convert a seven-dimensional PSO particle into valid XGBoost settings."""
    return {
        "n_estimators": int(np.clip(np.rint(position[0]), 50, 500)),
        "learning_rate": float(np.clip(position[1], 0.01, 0.30)),
        "subsample": float(np.clip(position[2], 0.50, 1.00)),
        "max_depth": int(np.clip(np.rint(position[3]), 2, 15)),
        "min_child_weight": float(np.clip(position[4], 1.0, 20.0)),
        "gamma": float(np.clip(position[5], 0.0, 12.0)),
        "colsample_bytree": float(np.clip(position[6], 0.30, 1.00)),
    }


def make_xgb_model(params: dict[str, Any], random_seed: int, n_jobs: int = 1) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=random_seed,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
    )


def optimize_xgb_with_pso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_particles: int,
    iterations: int,
    c1: float,
    c2: float,
    inertia: float,
    cv_folds: int,
    random_seed: int,
) -> tuple[dict[str, Any], float]:
    """A compact global-best PSO implementation that maximizes mean CV R²."""
    lower = np.array([50, 0.01, 0.50, 2, 1.0, 0.0, 0.30], dtype=float)
    upper = np.array([500, 0.30, 1.00, 15, 20.0, 12.0, 1.00], dtype=float)

    rng = np.random.default_rng(random_seed)
    positions = rng.uniform(lower, upper, size=(n_particles, len(lower)))
    velocities = rng.uniform(
        -0.10 * (upper - lower),
        0.10 * (upper - lower),
        size=positions.shape,
    )

    personal_best_positions = positions.copy()
    personal_best_costs = np.full(n_particles, np.inf)
    global_best_position = positions[0].copy()
    global_best_cost = np.inf

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    progress_bar = st.progress(0)
    progress_message = st.empty()
    best_message = st.empty()

    def evaluate_particle(position: np.ndarray) -> float:
        params = params_from_position(position)
        model = make_xgb_model(params, random_seed=random_seed, n_jobs=1)
        try:
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=splitter,
                scoring="r2",
                n_jobs=1,
                error_score=np.nan,
            )
            mean_score = float(np.nanmean(scores))
            if not np.isfinite(mean_score):
                return 1.0e9
            return -mean_score
        except Exception:
            return 1.0e9

    for iteration in range(iterations):
        costs = np.array([evaluate_particle(position) for position in positions])

        improved = costs < personal_best_costs
        personal_best_costs[improved] = costs[improved]
        personal_best_positions[improved] = positions[improved]

        iteration_best_index = int(np.argmin(costs))
        if costs[iteration_best_index] < global_best_cost:
            global_best_cost = float(costs[iteration_best_index])
            global_best_position = positions[iteration_best_index].copy()

        current_best_r2 = -global_best_cost
        completed = int(100 * (iteration + 1) / iterations)
        progress_bar.progress(completed)
        progress_message.write(
            f"PSO progress: **{iteration + 1}/{iterations} iterations** ({completed}%)"
        )
        best_message.write(f"Best cross-validation R²: **{current_best_r2:.4f}**")

        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            inertia * velocities
            + c1 * r1 * (personal_best_positions - positions)
            + c2 * r2 * (global_best_position - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)

    progress_message.success("PSO hyperparameter optimization completed.")
    return params_from_position(global_best_position), float(-global_best_cost)


def create_r2_plot(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    r2_train: float,
    r2_test: float,
    r2_all: float,
    model_name: str,
) -> tuple[plt.Figure, bytes]:
    y_all = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(
        y_train,
        y_train_pred,
        label="Train data",
        s=34,
        marker="o",
        facecolors="none",
        edgecolors="#E6A56A",
        linewidths=1.0,
        alpha=0.85,
    )
    ax.scatter(
        y_test,
        y_test_pred,
        label="Test data",
        s=36,
        marker="o",
        color="#423C9B",
        alpha=0.85,
    )

    minimum = float(min(np.min(y_all), np.min(y_all_pred)))
    maximum = float(max(np.max(y_all), np.max(y_all_pred)))
    span = maximum - minimum
    margin = 0.05 * span if span > 0 else 1.0
    low = minimum - margin
    high = maximum + margin

    ax.plot([low, high], [low, high], "--", color="#315BB5", linewidth=1.4, label="45° line")

    if len(np.unique(y_all)) >= 2:
        slope, intercept = np.polyfit(y_all, y_all_pred, 1)
        x_line = np.linspace(low, high, 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5, label="Best-fit line")

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Experimental Degradation or Removal (%)", fontsize=11)
    ax.set_ylabel("Predicted Degradation or Removal (%)", fontsize=11)
    ax.set_title(f"{model_name} Model", fontsize=14)
    ax.text(
        0.04,
        0.96,
        f"Train R² = {r2_train:.4f}\nTest R² = {r2_test:.4f}\nTotal R² = {r2_all:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="lower right")
    ax.grid(False)
    fig.tight_layout()

    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
    png_buffer.seek(0)
    return fig, png_buffer.getvalue()


def create_excel_results(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    params_df: pd.DataFrame,
    cv_results: dict[str, Any],
) -> bytes:
    """One downloadable workbook: original holdout + five-fold CV/SHAP/PDP."""
    from openpyxl.styles import Font, PatternFill

    output = io.BytesIO()
    sheets = {
        "Holdout Predictions": results_df,
        "Holdout Metrics": metrics_df,
        "Hyperparameters": params_df,
        "CV Fold Metrics": cv_results["fold_metrics"],
        "CV Summary": cv_results["summary"],
        "CV OOF Predictions": cv_results["oof"],
        "CV SHAP Importance": cv_results["shap_importance"],
        "CV SHAP Values": cv_results["shap_values"],
        "CV PDP Curves": cv_results["pdp"],
        "CV Methodology": pd.DataFrame({"Explanation": cv_results["explanation"]}),
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "B2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
                cell.fill = PatternFill(fill_type="solid", fgColor="17365D")
            # Reasonable, bounded widths; avoid huge fields making the sheet unusable.
            for col_cells in sheet.columns:
                first = col_cells[0]
                width = min(42, max(13, len(str(first.value or "")) + 2))
                sheet.column_dimensions[first.column_letter].width = width
    output.seek(0)
    return output.getvalue()


def create_model_download(model: XGBRegressor) -> bytes:
    output = io.BytesIO()
    joblib.dump(model, output)
    output.seek(0)
    return output.getvalue()


# ============================================================
# Five-fold cross-validation: out-of-fold SHAP and fold-model PDP
# ============================================================
def figure_png(fig: plt.Figure) -> bytes:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=230, bbox_inches="tight")
    plt.close(fig)
    return stream.getvalue()


def pdp_grid_for_feature(values: np.ndarray, categorical: bool, points: int) -> np.ndarray:
    unique = np.unique(values)
    if categorical or len(unique) <= points:
        return unique
    low, high = np.quantile(values, [0.05, 0.95])
    return np.linspace(low, high, points) if high > low else unique[:points]


def explain_five_fold_cv(
    X: np.ndarray,
    y: np.ndarray,
    excel_rows: np.ndarray,
    parameters: dict[str, Any],
    seed: int,
    *,
    do_explanations: bool,
    reference_size: int,
    grid_points: int,
    scope_label: str,
) -> dict[str, Any]:
    """Five-fold OOF predictions from the selected CV scope.

    The five outer fold models use already-selected hyperparameters (not nested
    tuning). The outer folds use random_state = seed + 1, so they differ from
    the folds used inside the PSO search (random_state = seed).
    SHAP is calculated ONLY for each model's validation rows, using a
    background sampled exclusively from that fold's training rows.
    PDP uses each fold's fitted model and its fold-training reference rows.
    Inputs that are constant in the CV data cannot affect the model; they keep
    a SHAP value of zero in the tables but are omitted from SHAP/PDP figures.
    """
    cv_seed = seed + 1
    splitter = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    oof_pred = np.full(len(y), np.nan)
    oof_fold = np.zeros(len(y), dtype=int)
    oof_shap = np.full((len(y), X.shape[1]), np.nan)
    base_values = np.full(len(y), np.nan)
    fold_metrics: list[dict[str, Any]] = []
    grids = [pdp_grid_for_feature(X[:, j], INPUT_COLUMNS[j] == "Light source", grid_points)
             for j in range(X.shape[1])]
    pdp_by_feature: list[list[np.ndarray]] = [[] for _ in INPUT_COLUMNS]
    varying = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0]
    constant_features = [INPUT_COLUMNS[j] for j in range(X.shape[1]) if j not in varying]

    if do_explanations:
        import shap

    for fold, (tr, va) in enumerate(splitter.split(X, y), start=1):
        model = make_xgb_model(parameters, random_seed=seed + fold, n_jobs=1)
        model.fit(X[tr], y[tr])
        pred_tr = model.predict(X[tr])
        pred_va = model.predict(X[va])
        oof_pred[va] = pred_va
        oof_fold[va] = fold
        for name, targets, predictions in (("Train", y[tr], pred_tr),
                                           ("Validation", y[va], pred_va)):
            fold_metrics.append({"Fold": fold, **calculate_metrics(targets, predictions, name)})

        if do_explanations:
            rng = np.random.default_rng(seed + 1000 + fold)
            bg_idx = rng.choice(tr, size=min(reference_size, len(tr)), replace=False)
            bg = X[bg_idx]
            try:
                explainer = shap.TreeExplainer(
                    model, data=bg, feature_perturbation="interventional",
                    model_output="raw",
                )
                vals = np.asarray(explainer.shap_values(X[va], check_additivity=True))
                baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
                if vals.shape != (len(va), X.shape[1]):
                    raise ValueError(f"SHAP output shape {vals.shape} is incorrect")
                # The SHAP units match model output (% degradation).
                if not np.allclose(baseline + vals.sum(axis=1), pred_va,
                                   rtol=1e-3, atol=0.03):
                    raise ValueError("SHAP decomposition does not reproduce held-out predictions")
            except Exception as exc:
                raise RuntimeError(
                    f"SHAP calculation failed in fold {fold}. Check installed SHAP/XGBoost "
                    f"compatibility. No model was retrained to bypass this error: {exc}"
                ) from exc
            oof_shap[va] = vals
            base_values[va] = baseline

            for j in varying:
                grid = grids[j]
                temp = X[tr].copy()  # train-only fold reference
                output = np.empty(len(grid))
                for i, val in enumerate(grid):
                    temp[:, j] = val
                    output[i] = float(np.mean(model.predict(temp)))
                pdp_by_feature[j].append(output)

    if not np.isfinite(oof_pred).all() or not np.all(oof_fold > 0):
        raise RuntimeError("Five-fold validation did not cover every training row exactly once")
    oof_df = pd.DataFrame({
        "Original Excel Row": excel_rows,
        "Training-subset Row": np.arange(1, len(y) + 1),
        "CV Fold": oof_fold,
        "Experimental (%)": y,
        "CV OOF Prediction (%)": oof_pred,
        "Residual (Exp - CV OOF)": y - oof_pred,
        "Absolute Error": np.abs(y - oof_pred),
    })
    for j, feature in enumerate(INPUT_COLUMNS):
        oof_df[feature] = X[:, j]
    fold_df = pd.DataFrame(fold_metrics)
    validation = fold_df.loc[fold_df["Dataset"] == "Validation"]
    pooled = calculate_metrics(y, oof_pred, "Pooled OOF (training subset)")
    summary_rows = []
    for metric in ("R2", "RMSE", "MAE", "MAPE (%)"):
        numbers = validation[metric].to_numpy(dtype=float)
        summary_rows.append({
            "Metric": metric,
            "Mean across validation folds": float(np.nanmean(numbers)),
            "SD across validation folds": float(np.nanstd(numbers, ddof=1)),
            "Pooled OOF": pooled[metric],
        })
    summary_df = pd.DataFrame(summary_rows)
    plots: dict[str, bytes] = {}
    fig, ax = plt.subplots(figsize=(6.8, 6.5))
    ax.scatter(y, oof_pred, s=22, alpha=0.75)
    bounds = [float(min(y.min(), oof_pred.min())), float(max(y.max(), oof_pred.max()))]
    ax.plot(bounds, bounds, "k--", linewidth=1)
    ax.set(xlabel="Experimental removal (%)", ylabel="Out-of-fold prediction (%)",
           title=f"Five-fold CV | pooled R² = {pooled['R2']:.3f}")
    fig.tight_layout()
    plots["CV5_OOF_scatter.png"] = figure_png(fig)

    explanation = [
        f"Five-fold KFold(shuffle=True, random_state={cv_seed}) on: {scope_label}. "
        "These outer folds are deliberately different from the folds used inside the PSO search.",
        "All five CV models use hyperparameters selected for the original holdout model. If PSO was used, its CV search used the original development subset; these results are post-selection diagnostics, NOT independent nested-CV estimates.",
        "When five-fold CV uses the FULL dataset, the external holdout rows are deliberately included in that separate CV experiment; full-dataset CV and the original holdout are NOT independent evidence.",
        "CV pooled metrics are based on exactly one out-of-fold validation prediction per development record. Fold mean/SD use the five separate validation folds.",
        "MAPE excludes observations with exactly zero experimental response; MAPE near zero removal can be unstable.",
        "SHAP values are computed using only the validation fold for each model; fold-specific interventional background rows are drawn only from each fold's own training data.",
        "PDP varies one feature at a time for each fold's model and averages predictions over that fold's training rows. Curves are descriptive, not causal; correlated input settings may form unrealistic combinations.",
        "Light source codes are 1=UV, 2=visible and 3=simulated solar; they are numeric encoded, so model splits can impose ordinal relations.",
        f"Random seed for outer five-fold CV: {cv_seed}; CV observations: {len(y)}; inputs: {len(INPUT_COLUMNS)}.",
    ]
    if constant_features:
        explanation.append(
            "Inputs constant in the CV data (they cannot affect predictions, have zero SHAP "
            "and are omitted from the SHAP and PDP figures): " + ", ".join(constant_features) + "."
        )
    shap_importance_df = pd.DataFrame(columns=[
        "SHAP Rank", "Feature", "Mean absolute OOF SHAP (percentage points)", "Varies in CV data"])
    shap_values_df = pd.DataFrame()
    pdp_df = pd.DataFrame()
    if do_explanations:
        if not np.isfinite(oof_shap).all():
            raise RuntimeError("Not every validation record received a SHAP explanation")
        importance = np.mean(np.abs(oof_shap), axis=0)
        order = np.argsort(-importance, kind="stable")
        shap_importance_df = pd.DataFrame({
            "SHAP Rank": np.arange(1, len(order) + 1),
            "Feature": [INPUT_COLUMNS[j] for j in order],
            "Mean absolute OOF SHAP (percentage points)": importance[order],
            "Varies in CV data": ["Yes" if j in varying else "No (constant)" for j in order],
        })
        shap_values_df = pd.DataFrame(oof_shap, columns=INPUT_COLUMNS)
        shap_values_df.insert(0, "Original Excel Row", excel_rows)
        shap_values_df.insert(1, "CV Fold", oof_fold)
        shap_values_df.insert(2, "SHAP baseline", base_values)
        shap_values_df.insert(3, "OOF Prediction", oof_pred)
        shap_values_df.insert(4, "Experimental", y)

        # Figures show only inputs that vary; ordered by mean absolute held-out SHAP.
        plot_order = [j for j in order if j in varying]
        names = [INPUT_COLUMNS[j] for j in plot_order]
        if plot_order:
            # SHAP beeswarm built from one out-of-fold explanation per record.
            plt.close("all")
            shap.summary_plot(oof_shap[:, plot_order], X[:, plot_order], feature_names=names,
                              max_display=len(plot_order), sort=False, show=False,
                              plot_size=(10, max(4.0, 0.55 * len(plot_order) + 2)))
            ax = plt.gca()
            ax.set_xlabel("Held-out SHAP contribution (percentage points)")
            plt.tight_layout()
            plots["CV5_SHAP_beeswarm.png"] = figure_png(plt.gcf())

            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(plot_order) + 1.5)))
            increasing = plot_order[::-1]
            ax.barh([INPUT_COLUMNS[j] for j in increasing], importance[increasing])
            ax.set_xlabel("Mean absolute OOF SHAP (percentage points)")
            ax.set_title("SHAP importance: five-fold out-of-fold predictions")
            fig.tight_layout()
            plots["CV5_SHAP_importance.png"] = figure_png(fig)

            rows = []
            nrows = math.ceil(len(plot_order) / 3)
            fig, axes = plt.subplots(nrows, 3, figsize=(16, nrows * 3.6))
            axes = np.asarray(axes).reshape(-1)
            for rank, j in enumerate(plot_order, start=1):
                grid = grids[j]
                arr = np.asarray(pdp_by_feature[j])
                mu = arr.mean(axis=0)
                for k, xi in enumerate(grid):
                    row = {"SHAP Rank": rank, "Feature": INPUT_COLUMNS[j],
                           "Feature Value": float(xi), "Mean PDP (%)": float(mu[k]),
                           "Min PDP (%)": float(arr[:, k].min()),
                           "Max PDP (%)": float(arr[:, k].max()),
                           "SD across folds": float(arr[:, k].std(ddof=1))}
                    row.update({f"Fold {fold} PDP (%)": float(arr[fold - 1, k])
                                for fold in range(1, 6)})
                    rows.append(row)
                ax = axes[rank - 1]
                for fold in range(5):
                    ax.plot(grid, arr[fold], alpha=0.65, lw=1, label=f"Fold {fold+1}",
                            marker="o" if len(grid) <= 6 else None, markersize=3)
                if len(grid) > 1:
                    ax.fill_between(grid, arr.min(axis=0), arr.max(axis=0), alpha=0.13)
                if INPUT_COLUMNS[j] == "Light source":
                    ax.set_xticks([1, 2, 3], ["UV", "Visible", "Solar"])
                ax.set_title(f"({chr(96 + rank)}) {INPUT_COLUMNS[j]}", fontsize=9)
                ax.set_xlabel("Feature value", fontsize=8)
                ax.set_ylabel("Predicted degradation (%)", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.2)
            for ax in axes[len(plot_order):]:
                ax.axis("off")
            axes[0].legend(fontsize=7, ncol=2)
            fig.suptitle("Five-fold PDP panels ordered by mean absolute held-out SHAP", y=1.005)
            fig.tight_layout()
            plots["CV5_PDP_SHAP_order.png"] = figure_png(fig)
            pdp_df = pd.DataFrame(rows)
    else:
        explanation.append("SHAP and PDP were disabled for this run; their workbook sheets are blank.")

    return {"fold_metrics": fold_df, "summary": summary_df,
            "oof": oof_df, "shap_importance": shap_importance_df,
            "shap_values": shap_values_df, "pdp": pdp_df,
            "plots": plots, "explanation": explanation,
            "constant_features": constant_features}


def zip_plots(plots: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, data in plots.items():
            archive.writestr(filename, data)
    return output.getvalue()


# ============================================================
# Step 1: Template download
# ============================================================
st.subheader("Step 1: Download the Excel Data Template")

st.download_button(
    label="Download SampleData.xlsx",
    data=get_template_bytes(),
    file_name="SampleData.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.info(
    "The workbook contains exactly 11 model inputs and one output. Use the final column, "
    f"'{OUTPUT_COLUMN}', for the measured experimental result. Light-source coding: "
    "1 = UV, 2 = visible light, and 3 = simulated solar light. Text labels are also accepted."
)


# ============================================================
# Step 2: Upload and validate dataset
# ============================================================
st.subheader("Step 2: Upload the Completed Excel Dataset")

uploaded_file = st.file_uploader(
    "Upload the completed SampleData.xlsx file",
    type=["xlsx"],
)

if uploaded_file is None:
    st.warning("Download the template, enter the complete dataset, and upload the completed Excel file.")
    st.stop()

uploaded_bytes = uploaded_file.getvalue()
current_signature = hashlib.sha256(uploaded_bytes).hexdigest()
if st.session_state.dataset_signature != current_signature:
    clear_model_results()
    st.session_state.dataset_signature = current_signature

try:
    raw_df = pd.read_excel(io.BytesIO(uploaded_bytes))
except Exception as exc:
    st.error(f"The uploaded Excel file could not be read: {exc}")
    st.stop()

# Match headers regardless of case, extra spaces or look-alike Unicode characters.
raw_df, duplicate_columns = standardise_columns(raw_df)
if duplicate_columns:
    st.error(
        "The uploaded workbook contains more than one column for: "
        + ", ".join(duplicate_columns)
        + ". Keep only one column for each variable."
    )
    st.stop()

missing_columns = [column for column in TEMPLATE_COLUMNS if column not in raw_df.columns]
if missing_columns:
    st.error("The uploaded workbook does not match the required 11-input/1-output template.")
    st.write("Missing columns:")
    st.code("\n".join(missing_columns))
    st.write("Required columns in order:")
    st.dataframe(pd.DataFrame({"Required column": TEMPLATE_COLUMNS}), width="stretch")
    st.stop()

working_df = raw_df[TEMPLATE_COLUMNS].copy()
working_df["Light source"] = encode_light_source(working_df["Light source"])
for column in TEMPLATE_COLUMNS:
    if column != "Light source":
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

invalid_light = ~working_df["Light source"].isin([1.0, 2.0, 3.0])
working_df.loc[invalid_light, "Light source"] = np.nan

rows_before = len(working_df)
working_df["Original Excel Row"] = np.arange(len(working_df)) + 2
working_df = working_df.replace([np.inf, -np.inf], np.nan)
working_df = working_df.dropna(subset=TEMPLATE_COLUMNS).reset_index(drop=True)
removed_rows = rows_before - len(working_df)

if len(working_df) < 20:
    st.error(
        "At least 20 complete numerical rows are required for five-fold CV with the selected holdout split. "
        f"Only {len(working_df)} valid rows were detected."
    )
    st.stop()

constant_inputs = [c for c in INPUT_COLUMNS if working_df[c].nunique() <= 1]
if len(constant_inputs) == len(INPUT_COLUMNS):
    st.error("All 11 inputs have a single value in this dataset, so no model can be developed.")
    st.stop()

st.success(f"Excel dataset uploaded successfully: {len(working_df)} valid rows.")
if removed_rows:
    st.warning(f"{removed_rows} incomplete or invalid rows were excluded before modelling.")
if constant_inputs:
    st.info(
        f"{len(constant_inputs)} input(s) have a single value in this dataset and cannot "
        "influence the predictions: " + ", ".join(constant_inputs)
        + ". They remain in the model input, but are omitted from the SHAP and PDP figures."
    )

with st.expander("Preview the validated modelling dataset", expanded=True):
    st.dataframe(working_df.head(20), width="stretch")


# ============================================================
# Step 3: Model settings
# ============================================================
st.subheader("Step 3: Select the XGBoost–PSO Model Settings")

train_percent = st.slider(
    "Training data percentage (%)",
    min_value=50,
    max_value=90,
    value=85,
    step=5,
)
test_percent = 100 - train_percent
st.write(f"Selected data division: **{train_percent}% training / {test_percent}% testing**")

estimated_test_rows = int(np.ceil(len(working_df) * test_percent / 100.0))
if estimated_test_rows < 2:
    st.error("The selected split produces fewer than two test observations. Reduce the training percentage.")
    st.stop()

st.sidebar.header("Model Development Settings")
use_pso = st.sidebar.checkbox("Use PSO hyperparameter optimization", value=True)
if use_pso:
    st.sidebar.subheader("PSO Settings")
    n_particles = st.sidebar.slider("Number of particles", 5, 30, 5, 5)
    pso_iterations = st.sidebar.slider("PSO iterations", 5, 100, 30, 5)
    c1 = st.sidebar.number_input("Cognitive coefficient (c1)", 0.1, 5.0, 1.5, 0.1)
    c2 = st.sidebar.number_input("Social coefficient (c2)", 0.1, 5.0, 1.5, 0.1)
    inertia = st.sidebar.number_input("Inertia weight (w)", 0.1, 1.5, 0.7, 0.1)
    cv_folds_requested = st.sidebar.slider("Cross-validation folds", 2, 10, 5, 1)
    st.sidebar.caption(
        "The manuscript settings were 5 particles, 100 iterations, c1 = 1.5, "
        "c2 = 1.5 and w = 0.7. Fewer iterations run faster on Streamlit Cloud."
    )
else:
    st.sidebar.subheader("Manual XGBoost Hyperparameters")
    n_estimators = st.sidebar.slider("n_estimators", 50, 1000, 168, 10)
    learning_rate = st.sidebar.slider("learning_rate", 0.001, 0.300, 0.193, 0.001, format="%.3f")
    subsample = st.sidebar.slider("subsample", 0.50, 1.00, 0.718, 0.01)
    max_depth = st.sidebar.slider("max_depth", 1, 20, 9, 1)
    min_child_weight = st.sidebar.slider("min_child_weight", 0.1, 20.0, 4.7, 0.1)
    gamma = st.sidebar.slider("gamma", 0.0, 15.0, 8.1, 0.1)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.30, 1.00, 0.945, 0.005)


st.sidebar.divider()
st.sidebar.subheader("Five-fold CV and explainability")
cv_scope = st.sidebar.radio(
    "Select data for five-fold CV",
    options=["Training subset only", "Entire uploaded dataset"],
    index=0,
    help="Training-only CV keeps the original test records out of CV. Full-data CV uses every valid uploaded row, including holdout rows, and is a separate descriptive validation run.",
)

seed_setting = st.sidebar.number_input(
    "Base random seed", min_value=0, max_value=9999999, value=42, step=1,
    help="Attempt 1 uses this seed. Each further Develop/Redevelop click adds 1, giving a "
         "new train/test split and new CV folds. Re-entering the same settings reproduces "
         "the same sequence of results.",
)
do_explanations = st.sidebar.checkbox("Generate 5-fold SHAP and PDP", value=True)
if do_explanations:
    reference_size = st.sidebar.slider("SHAP background records per fold", 20, 150, 60, 10)
    grid_points = st.sidebar.slider("PDP grid points", 15, 80, 40, 5)
else:
    reference_size = 60
    grid_points = 40

# Invalidate displayed results when settings have changed since fitting.
settings = {"train_percent": train_percent, "use_pso": use_pso,
            "seed": int(seed_setting), "cv_scope": cv_scope,
            "do_explanations": do_explanations,
            "reference_size": reference_size, "grid_points": grid_points}
if use_pso:
    settings.update({"n_particles": n_particles, "iterations": pso_iterations,
                     "c1": float(c1), "c2": float(c2), "inertia": float(inertia),
                     "search_cv_folds": cv_folds_requested})
else:
    settings.update({k: float(v) for k, v in {
        "n_estimators": n_estimators, "learning_rate": learning_rate,
        "subsample": subsample, "max_depth": max_depth,
        "min_child_weight": min_child_weight, "gamma": gamma,
        "colsample_bytree": colsample_bytree}.items()})
settings_signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
if (st.session_state.settings_signature is not None and
        st.session_state.settings_signature != settings_signature):
    clear_model_results()
st.session_state.settings_signature = settings_signature

if cv_scope == "Training subset only":
    st.info("The external test set is excluded from five-fold CV. The five-fold CV uses "
            "different folds from the PSO search, but the hyperparameters have already been "
            "selected by PSO, so the five-fold results are post-selection diagnostics rather "
            "than unbiased nested cross-validation.")
else:
    st.warning("Five-fold CV will use ALL uploaded records, including the original holdout rows. "
               "Treat it as a separate five-fold analysis rather than independent confirmation "
               "of the external holdout score. PSO settings are not retuned within each fold.")


# ============================================================
# Step 4: Develop model
# ============================================================
st.subheader("Step 4: Develop or Redevelop the XGBoost–PSO Model")
st.caption(f"The next run will use random seed {int(seed_setting) + st.session_state.attempt} "
           "(a new train/test split each time you click).")

run_model = st.button("Develop / Redevelop XGBoost–PSO Model", type="primary")

if run_model:
    st.session_state.attempt += 1
    # Each Develop/Redevelop click uses a new, reproducible seed:
    # attempt 1 = base seed, attempt 2 = base seed + 1, and so on.
    random_seed = int(seed_setting) + st.session_state.attempt - 1

    X = working_df[INPUT_COLUMNS].to_numpy(dtype=float)
    y = working_df[OUTPUT_COLUMN].to_numpy(dtype=float)
    row_indices = np.arange(len(working_df))

    try:
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X,
            y,
            row_indices,
            train_size=train_percent / 100.0,
            random_state=random_seed,
            shuffle=True,
        )
    except Exception as exc:
        st.error(f"The train/test split could not be created: {exc}")
        st.stop()

    if len(y_test) < 2:
        st.error("The test subset contains fewer than two observations; R² cannot be calculated.")
        st.stop()

    if len(y_train) < 10:
        st.error("Five-fold CV needs at least 10 training rows (at least 2 held-out rows per fold).")
        st.stop()

    best_cv_r2 = float("nan")

    with st.spinner("Developing the model. This may take several minutes when PSO is enabled..."):
        if use_pso:
            # R² requires at least two observations in each validation fold.
            effective_cv = min(cv_folds_requested, max(2, len(y_train) // 2))
            if effective_cv < 2:
                st.error("Insufficient training observations for cross-validation.")
                st.stop()
            best_params, best_cv_r2 = optimize_xgb_with_pso(
                X_train,
                y_train,
                n_particles=n_particles,
                iterations=pso_iterations,
                c1=float(c1),
                c2=float(c2),
                inertia=float(inertia),
                cv_folds=effective_cv,
                random_seed=random_seed,
            )
            model_name = "XGBoost–PSO"
        else:
            best_params = {
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "subsample": float(subsample),
                "max_depth": int(max_depth),
                "min_child_weight": float(min_child_weight),
                "gamma": float(gamma),
                "colsample_bytree": float(colsample_bytree),
            }
            model_name = "XGBoost"

        model = make_xgb_model(best_params, random_seed=random_seed, n_jobs=-1)
        model.fit(X_train, y_train)

    # Separate five-fold CV. Five fold-specific models explain validation rows.
    with st.spinner("Calculating five-fold CV predictions, SHAP values and PDPs..."):
        try:
            if cv_scope == "Training subset only":
                cv_indices = train_indices
                cv_X, cv_y = X_train, y_train
            else:
                cv_indices = row_indices
                cv_X, cv_y = X, y
            cv_results = explain_five_fold_cv(
                cv_X, cv_y,
                working_df.loc[cv_indices, "Original Excel Row"].to_numpy(),
                best_params, random_seed, do_explanations=do_explanations,
                reference_size=reference_size, grid_points=grid_points,
                scope_label=cv_scope,
            )
        except Exception as exc:
            st.error(f"Five-fold cross-validation / explainability failed: {exc}")
            st.stop()

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(X)

    train_metrics = calculate_metrics(y_train, y_train_pred, "Train")
    test_metrics = calculate_metrics(y_test, y_test_pred, "Test")
    total_metrics = calculate_metrics(y, y_all_pred, "Total")
    metrics_df = pd.DataFrame([train_metrics, test_metrics, total_metrics])

    results_df = working_df.copy()
    results_df["Predicted Degradation or Removal (%)"] = y_all_pred
    results_df["Residual (Experimental - Predicted)"] = y - y_all_pred
    results_df["Absolute Error"] = np.abs(y - y_all_pred)
    results_df["Data Set"] = ""
    results_df.loc[train_indices, "Data Set"] = "Train"
    results_df.loc[test_indices, "Data Set"] = "Test"

    results_df["CV Fold"] = np.nan
    results_df["CV OOF Prediction (%)"] = np.nan
    results_df.loc[cv_indices, "CV Fold"] = cv_results["oof"]["CV Fold"].to_numpy()
    results_df.loc[cv_indices, "CV OOF Prediction (%)"] = cv_results["oof"]["CV OOF Prediction (%)"].to_numpy()

    pso_settings = {
        "Use PSO": use_pso,
        "Particles": n_particles if use_pso else None,
        "PSO iterations": pso_iterations if use_pso else None,
        "c1": float(c1) if use_pso else None,
        "c2": float(c2) if use_pso else None,
        "w": float(inertia) if use_pso else None,
        "CV folds": effective_cv if use_pso else None,
        "Best CV R2": best_cv_r2 if use_pso else None,
    }
    params_df = pd.DataFrame(
        [
            {
                **best_params,
                **pso_settings,
                "Training percentage": train_percent,
                "Testing percentage": test_percent,
                "Total valid rows": len(working_df),
                "Base random seed": int(seed_setting),
                "Random seed used": random_seed,
                "Attempt": st.session_state.attempt,
            }
        ]
    )

    fig, plot_png = create_r2_plot(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        train_metrics["R2"],
        test_metrics["R2"],
        total_metrics["R2"],
        model_name,
    )
    plt.close(fig)

    excel_results = create_excel_results(
        results_df,
        metrics_df,
        params_df,
        cv_results,
    )
    model_file = create_model_download(model)

    st.session_state.model_trained = True
    st.session_state.model = model
    st.session_state.results_df = results_df
    st.session_state.metrics_df = metrics_df
    st.session_state.params_df = params_df
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.y_train_pred = y_train_pred
    st.session_state.y_test_pred = y_test_pred
    st.session_state.plot_png = plot_png
    st.session_state.excel_results = excel_results
    st.session_state.model_file = model_file
    st.session_state.best_cv_r2 = best_cv_r2
    st.session_state.model_name = model_name
    st.session_state.random_seed_used = random_seed
    st.session_state.cv_metrics_df = cv_results["fold_metrics"]
    st.session_state.cv_summary_df = cv_results["summary"]
    st.session_state.cv_oof_df = cv_results["oof"]
    st.session_state.cv_shap_importance_df = cv_results["shap_importance"]
    st.session_state.cv_shap_values_df = cv_results["shap_values"]
    st.session_state.cv_pdp_df = cv_results["pdp"]
    st.session_state.cv_plots = cv_results["plots"]
    st.session_state.cv_plots_zip = zip_plots(cv_results["plots"])
    st.session_state.cv_explanation = cv_results["explanation"]


# ============================================================
# Results remain visible after download-button reruns
# ============================================================
if st.session_state.model_trained:
    metrics_df = st.session_state.metrics_df
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

    st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.subheader("Experimental versus Predicted Plot")
    st.image(st.session_state.plot_png)

    st.subheader("Optimized Model Hyperparameters")
    st.dataframe(st.session_state.params_df, width="stretch", hide_index=True)

    with st.expander("Prediction results", expanded=False):
        st.dataframe(st.session_state.results_df, width="stretch")

    st.divider()
    st.subheader("Five-Fold Cross-Validation Results")
    st.caption(f"CV scope: **{cv_scope}**. Predictions are out-of-fold for this selected scope. "
               "PSO-tuned five-fold scores are post-selection diagnostics, not nested CV estimates.")
    st.dataframe(st.session_state.cv_summary_df, hide_index=True, width="stretch")
    with st.expander("Fold-by-fold train and validation metrics"):
        st.dataframe(st.session_state.cv_metrics_df, hide_index=True, width="stretch")
    with st.expander("Out-of-fold predictions and original Excel row numbers"):
        st.dataframe(st.session_state.cv_oof_df, hide_index=True, width="stretch")
    st.image(st.session_state.cv_plots["CV5_OOF_scatter.png"], caption="Five-fold CV out-of-fold predictions")
    if "CV5_SHAP_beeswarm.png" in st.session_state.cv_plots:
        st.subheader("Five-Fold Out-of-Fold SHAP Analysis")
        st.caption("Each record is explained only using the model that did not train on it. "
                   "The SHAP background is drawn from that model's training fold. "
                   "Inputs that are constant in the data are omitted from the figures.")
        st.image(st.session_state.cv_plots["CV5_SHAP_beeswarm.png"])
        st.image(st.session_state.cv_plots["CV5_SHAP_importance.png"])
        st.dataframe(st.session_state.cv_shap_importance_df, hide_index=True,
                     width="stretch")
        st.subheader("Five-Fold Partial Dependence Plots")
        st.caption("PDP panels appear from highest to lowest mean absolute held-out SHAP importance. "
                   "Each curve represents one of the five fold-specific models.")
        st.image(st.session_state.cv_plots["CV5_PDP_SHAP_order.png"])
        with st.expander("Numerical PDP data (also in the Excel file)"):
            st.dataframe(st.session_state.cv_pdp_df, hide_index=True, width="stretch")
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
