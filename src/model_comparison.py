from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

DATA_FILE = REPORTS / "multimodal_window_features.csv"

REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Columns that identify a row/window or contain the target/derived target.
NON_FEATURE_COLUMNS = {
    "match_id",
    "player_folder",
    "group_id",
    "window_start",
    "window_end",
    "stress_target",
    "stress_class",
}


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. "
            "Run build_multimodal_dataset.py first."
        )

    df = pd.read_csv(DATA_FILE)

    if "stress_target" not in df.columns:
        raise ValueError("stress_target column is missing.")

    if "player_folder" not in df.columns:
        raise ValueError("player_folder column is missing.")

    X = df.drop(columns=list(NON_FEATURE_COLUMNS), errors="ignore")
    X = X.select_dtypes(include=np.number).copy()
    y = pd.to_numeric(df["stress_target"], errors="coerce")
    groups = df["player_folder"].astype(str)

    valid = y.notna() & groups.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    groups = groups.loc[valid].reset_index(drop=True)
    df = df.loc[valid].reset_index(drop=True)

    return df, X, y, groups


def build_models():
    """
    Models are deliberately chosen to represent different assumptions:
    - Dummy: no-learning baseline
    - Ridge: linear relationship
    - Random Forest: nonlinear bagged trees
    - Extra Trees: randomized tree ensemble
    - HistGradientBoosting: boosted nonlinear model
    - SVR: nonlinear kernel model
    """

    return {
        "Dummy Mean": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]),

        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        "Hist Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )),
        ]),

        "SVR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVR(
                C=1.0,
                epsilon=0.03,
                kernel="rbf",
            )),
        ]),
    }


def evaluate_models(df, X, y, groups):
    logo = LeaveOneGroupOut()

    summary_rows = []
    fold_rows = []
    prediction_rows = []

    models = build_models()

    print("=" * 70)
    print("MODEL COMPARISON — LEAVE-ONE-PLAYER-OUT")
    print("=" * 70)
    print(f"Rows: {len(df)}")
    print(f"Predictors: {X.shape[1]}")
    print(f"Players: {sorted(groups.unique().tolist())}")
    print()

    for model_name, model in models.items():

        print(f"Evaluating: {model_name}")

        model_predictions = []

        for fold, (train_idx, test_idx) in enumerate(
            logo.split(X, y, groups), start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            test_group = groups.iloc[test_idx].iloc[0]

            # Fit ONLY on the training players.
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, pred)
            rmse = mean_squared_error(y_test, pred) ** 0.5
            r2 = r2_score(y_test, pred)

            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "test_player": test_group,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            })

            for row_idx, actual, predicted in zip(
                test_idx, y_test, pred
            ):
                prediction_rows.append({
                    "model": model_name,
                    "row_index": int(row_idx),
                    "player_folder": groups.iloc[row_idx],
                    "match_id": df.iloc[row_idx]["match_id"],
                    "window_start": df.iloc[row_idx]["window_start"],
                    "window_end": df.iloc[row_idx]["window_end"],
                    "actual_stress": float(actual),
                    "predicted_stress": float(predicted),
                })

        model_folds = pd.DataFrame([
            r for r in fold_rows if r["model"] == model_name
        ])

        summary_rows.append({
            "model": model_name,
            "MAE_mean": model_folds["MAE"].mean(),
            "MAE_std": model_folds["MAE"].std(),
            "RMSE_mean": model_folds["RMSE"].mean(),
            "RMSE_std": model_folds["RMSE"].std(),
            "R2_mean": model_folds["R2"].mean(),
            "R2_std": model_folds["R2"].std(),
        })

    summary = pd.DataFrame(summary_rows)

    # Primary selection criterion: lowest mean MAE.
    # RMSE and R2 are retained for interpretation.
    summary = summary.sort_values(
        ["MAE_mean", "RMSE_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary.to_csv(
        REPORTS / "model_comparison.csv",
        index=False,
    )

    folds.to_csv(
        REPORTS / "model_comparison_folds.csv",
        index=False,
    )

    predictions.to_csv(
        REPORTS / "model_oof_predictions.csv",
        index=False,
    )

    selected_model_name = summary.iloc[0]["model"]

    # Train the selected model on all available players ONLY AFTER
    # the out-of-fold evaluation is complete.
    final_model = build_models()[selected_model_name]
    final_model.fit(X, y)

    safe_name = (
        selected_model_name.lower()
        .replace(" ", "_")
    )

    joblib.dump(
        final_model,
        MODELS / f"selected_{safe_name}.joblib",
    )

    results = {
        "selected_model_by_mean_MAE": selected_model_name,
        "selection_rule": (
            "Lowest mean Leave-One-Player-Out MAE; "
            "RMSE used as secondary criterion."
        ),
        "rows": int(len(df)),
        "predictors": int(X.shape[1]),
        "players": sorted(groups.unique().tolist()),
        "comparison": summary.to_dict(orient="records"),
    }

    (REPORTS / "model_selection.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    display = summary.copy()

    for col in display.columns:
        if col not in {"model"}:
            display[col] = display[col].round(6)

    print(display.to_string(index=False))

    print()
    print("=" * 70)
    print(f"SELECTED MODEL: {selected_model_name}")
    print("=" * 70)

    print()
    print("Saved:")
    print(f"  {REPORTS / 'model_comparison.csv'}")
    print(f"  {REPORTS / 'model_comparison_folds.csv'}")
    print(f"  {REPORTS / 'model_oof_predictions.csv'}")
    print(f"  {REPORTS / 'model_selection.json'}")
    print(f"  {MODELS / f'selected_{safe_name}.joblib'}")


def main():
    df, X, y, groups = load_dataset()
    evaluate_models(df, X, y, groups)


if __name__ == "__main__":
    main()
