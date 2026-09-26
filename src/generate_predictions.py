from pathlib import Path
import json
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = ROOT / "reports"

OOF_FILE = REPORTS_DIR / "model_oof_predictions.csv"
OUTPUT_FILE = REPORTS_DIR / "stress_predictions.csv"
SELECTION_FILE = REPORTS_DIR / "model_selection.json"


def main():

    print("=" * 70)
    print("GENERATING OUT-OF-FOLD STRESS PREDICTIONS")
    print("=" * 70)

    if not OOF_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {OOF_FILE}"
        )

    df = pd.read_csv(OOF_FILE)

    print(f"Total OOF rows: {len(df)}")
    print(f"Models present: {df['model'].unique().tolist()}")

    # ------------------------------------------------------------
    # Load selected model
    # ------------------------------------------------------------

    selected_model = None

    if SELECTION_FILE.exists():

        with open(SELECTION_FILE, "r") as f:
            selection = json.load(f)

        selected_model = selection.get("selected_model")

    if selected_model is None:
        selected_model = "Extra Trees"

    print(f"Selected model: {selected_model}")

    # ------------------------------------------------------------
    # Keep only selected model
    # ------------------------------------------------------------

    pred = df[df["model"] == selected_model].copy()

    if pred.empty:
        raise ValueError(
            f"No OOF predictions found for model: {selected_model}"
        )

    # ------------------------------------------------------------
    # Rename columns
    # ------------------------------------------------------------

    rename_map = {
        "actual": "actual_stress",
        "predicted": "predicted_stress",
    }

    pred = pred.rename(columns=rename_map)

    # ------------------------------------------------------------
    # Make sure numeric columns are numeric
    # ------------------------------------------------------------

    pred["actual_stress"] = pd.to_numeric(
        pred["actual_stress"],
        errors="coerce"
    )

    pred["predicted_stress"] = pd.to_numeric(
        pred["predicted_stress"],
        errors="coerce"
    )

    pred["window_start"] = pd.to_numeric(
        pred["window_start"],
        errors="coerce"
    )

    pred["window_end"] = pd.to_numeric(
        pred["window_end"],
        errors="coerce"
    )

    pred = pred.dropna(
        subset=[
            "actual_stress",
            "predicted_stress",
            "window_start",
            "window_end"
        ]
    )

    # ------------------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------------------

    pred = pred.sort_values(
        ["player_folder", "match_id", "window_start"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Player-specific baseline
    #
    # IMPORTANT:
    # The baseline uses only previous windows.
    # Current/future stress is never used to calculate it.
    # ------------------------------------------------------------

    pred["stress_baseline"] = (
        pred.groupby(["player_folder", "match_id"])["predicted_stress"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(window=10, min_periods=3)
            .median()
        )
    )

    # For the first few windows, use the expanding
    # past-only median.

    def past_only_baseline(series):

        shifted = series.shift(1)

        return shifted.expanding(
            min_periods=1
        ).median()

    fallback = (
        pred.groupby(
            ["player_folder", "match_id"]
        )["predicted_stress"]
        .transform(past_only_baseline)
    )

    pred["stress_baseline"] = (
        pred["stress_baseline"]
        .fillna(fallback)
    )

    # ------------------------------------------------------------
    # Stress deviation
    # ------------------------------------------------------------

    pred["stress_deviation"] = (
        pred["predicted_stress"]
        - pred["stress_baseline"]
    )

    # ------------------------------------------------------------
    # Stress ratio
    # ------------------------------------------------------------

    denominator = pred["stress_baseline"].replace(0, np.nan)

    pred["stress_ratio"] = (
        pred["predicted_stress"] / denominator
    )

    # ------------------------------------------------------------
    # Player-relative thresholds
    #
    # These are calculated from each player's predicted
    # stress deviations.
    # ------------------------------------------------------------

    def assign_states(group):

        q50 = group["stress_deviation"].median()
        q75 = group["stress_deviation"].quantile(0.75)

        conditions = [
             group["stress_deviation"].isna(),
	         group["stress_deviation"] <= q50,
             group["stress_deviation"] <= q75,
        ]

        choices = [
            "Insufficient Baseline",
            "Normal",
            "Elevated",
        ]

        return np.select(
            conditions,
            choices,
            default="High"
        )

    pred["stress_state"] = (
        pred.groupby("player_folder", group_keys=False)
        .apply(
            lambda g: pd.Series(
                assign_states(g),
                index=g.index
            )
        )
        .sort_index()
    )

    # ------------------------------------------------------------
    # Clean model column
    # ------------------------------------------------------------

    pred["selected_model"] = selected_model

    # ------------------------------------------------------------
    # Select final columns
    # ------------------------------------------------------------

    final_columns = [
        "selected_model",
        "player_folder",
        "match_id",
        "window_start",
        "window_end",
        "actual_stress",
        "predicted_stress",
        "stress_baseline",
        "stress_deviation",
        "stress_ratio",
        "stress_state",
    ]

    pred = pred[final_columns]

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    pred.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("PREDICTION DATASET CREATED")
    print("=" * 70)

    print(f"Rows: {len(pred)}")
    print(
        f"Players: "
        f"{pred['player_folder'].nunique()}"
    )
    print(
        f"Matches: "
        f"{pred['match_id'].nunique()}"
    )

    print()
    print("Stress states:")

    print(
        pred["stress_state"]
        .value_counts()
        .to_string()
    )

    print()
    print("Prediction range:")
    print(
        f"Actual:    "
        f"{pred['actual_stress'].min():.4f} → "
        f"{pred['actual_stress'].max():.4f}"
    )

    print(
        f"Predicted: "
        f"{pred['predicted_stress'].min():.4f} → "
        f"{pred['predicted_stress'].max():.4f}"
    )

    print()
    print(f"Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()