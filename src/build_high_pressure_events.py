from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "reports"

PREDICTIONS_FILE = REPORTS / "stress_predictions.csv"
EVENTS_FILE = REPORTS / "gameplay_events.csv"
OUTPUT_FILE = REPORTS / "high_pressure_events.csv"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Sensor windows are 30 seconds.
# We consider an event associated with a stress window when
# the event falls inside that window.

BEFORE_SECONDS = 30
AFTER_SECONDS = 60

# A high-pressure-associated event requires stress deviation
# to be at or above the player's 75th percentile.
HIGH_PRESSURE_QUANTILE = 0.75


def find_column(df, candidates, description):
    """
    Find the first matching column from a list of candidates.
    Matching is case-insensitive.
    """

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def main():

    print("=" * 70)
    print("HIGH-PRESSURE EVENT ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{PREDICTIONS_FILE}"
        )

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{EVENTS_FILE}"
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)
    events = pd.read_csv(EVENTS_FILE)

    print(f"Prediction rows: {len(predictions)}")
    print(f"Gameplay events: {len(events)}")

    print()
    print("Prediction columns:")
    print(predictions.columns.tolist())

    print()
    print("Gameplay event columns:")
    print(events.columns.tolist())

    # --------------------------------------------------------
    # Identify event columns
    # --------------------------------------------------------

    event_player_col = find_column(
        events,
        [
            "player_folder",
            "player",
            "player_id",
            "participant",
        ],
        "player"
    )

    event_match_col = find_column(
        events,
        [
            "match_id",
            "match",
            "game_id",
        ],
        "match"
    )

    event_time_col = find_column(
        events,
        [
            "event_time",
            "timestamp",
            "time",
            "game_time",
            "time_seconds",
        ],
        "event timestamp"
    )

    event_type_col = find_column(
        events,
        [
            "event_type",
            "type",
            "event",
        ],
        "event type"
    )

    # --------------------------------------------------------
    # Standardize names
    # --------------------------------------------------------

    events = events.rename(
        columns={
            event_player_col: "player_folder",
            event_match_col: "match_id",
            event_time_col: "event_time",
            event_type_col: "event_type",
        }
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    predictions["window_start"] = pd.to_numeric(
        predictions["window_start"],
        errors="coerce"
    )

    predictions["window_end"] = pd.to_numeric(
        predictions["window_end"],
        errors="coerce"
    )

    predictions["predicted_stress"] = pd.to_numeric(
        predictions["predicted_stress"],
        errors="coerce"
    )

    predictions["stress_baseline"] = pd.to_numeric(
        predictions["stress_baseline"],
        errors="coerce"
    )

    predictions["stress_deviation"] = pd.to_numeric(
        predictions["stress_deviation"],
        errors="coerce"
    )

    events["event_time"] = pd.to_numeric(
        events["event_time"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    predictions = predictions.dropna(
        subset=[
            "player_folder",
            "match_id",
            "window_start",
            "window_end",
            "predicted_stress",
            "stress_baseline",
            "stress_deviation",
        ]
    ).copy()

    events = events.dropna(
        subset=[
            "player_folder",
            "match_id",
            "event_time",
            "event_type",
        ]
    ).copy()


    # --------------------------------------------------------
    # Normalize identifiers
    # --------------------------------------------------------

    predictions["player_folder"] = (
        predictions["player_folder"].astype(str)
    )

    predictions["match_id"] = (
        predictions["match_id"].astype(str)
    )

    events["player_folder"] = (
        "player_"
        + pd.to_numeric(
            events["player_folder"],
            errors="coerce"
        ).astype("Int64").astype(str)
    )

    events["match_id"] = (
        events["match_id"].astype(str)
    )

    # --------------------------------------------------------
    # Calculate player's high-pressure threshold
    #
    # This is a descriptive threshold based on predicted
    # stress deviations.
    # --------------------------------------------------------

    player_thresholds = (
        predictions
        .groupby("player_folder")["stress_deviation"]
        .quantile(HIGH_PRESSURE_QUANTILE)
        .rename("high_pressure_threshold")
        .reset_index()
    )

    predictions = predictions.merge(
        player_thresholds,
        on="player_folder",
        how="left"
    )

    # --------------------------------------------------------
    # Match each gameplay event to the sensor window that
    # contains the event.
    # --------------------------------------------------------

    records = []

    for _, event in events.iterrows():

        player = event["player_folder"]
        match = event["match_id"]
        event_time = float(event["event_time"])
        event_type = str(event["event_type"])

        player_windows = predictions[
            (predictions["player_folder"] == player)
            &
            (predictions["match_id"] == match)
        ].copy()

        if player_windows.empty:
            continue

        # Window containing the event.
        containing = player_windows[
            (player_windows["window_start"] <= event_time)
            &
            (player_windows["window_end"] > event_time)
        ]

        if containing.empty:
            continue

        current = containing.iloc[0]

        # ----------------------------------------------------
        # Before-event windows
        # ----------------------------------------------------

        before = player_windows[
            (
                player_windows["window_end"]
                <= event_time
            )
            &
            (
                player_windows["window_end"]
                > event_time - BEFORE_SECONDS
            )
        ]

        # ----------------------------------------------------
        # After-event windows
        # ----------------------------------------------------

        after = player_windows[
            (
                player_windows["window_start"]
                >= event_time
            )
            &
            (
                player_windows["window_start"]
                < event_time + AFTER_SECONDS
            )
        ]

        # ----------------------------------------------------
        # Extract stress values
        # ----------------------------------------------------

        stress_at_event = current["predicted_stress"]

        baseline_at_event = current["stress_baseline"]

        deviation_at_event = current["stress_deviation"]

        threshold = current["high_pressure_threshold"]

        # Before stress
        if not before.empty:
            stress_before = before[
                "predicted_stress"
            ].mean()

            deviation_before = before[
                "stress_deviation"
            ].mean()

        else:
            stress_before = np.nan
            deviation_before = np.nan

        # After stress
        if not after.empty:
            stress_after = after[
                "predicted_stress"
            ].mean()

            deviation_after = after[
                "stress_deviation"
            ].mean()

        else:
            stress_after = np.nan
            deviation_after = np.nan

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        immediate_change = (
            stress_at_event - stress_before
            if pd.notna(stress_before)
            else np.nan
        )

        after_change = (
            stress_after - stress_at_event
            if pd.notna(stress_after)
            else np.nan
        )

        # ----------------------------------------------------
        # High-pressure decision
        #
        # Event must have predicted stress deviation at or
        # above the player's 75th percentile.
        # ----------------------------------------------------

        is_high_pressure = (
            pd.notna(deviation_at_event)
            and
            pd.notna(threshold)
            and
            deviation_at_event >= threshold
        )

        records.append(
            {
                "player_folder": player,
                "match_id": match,
                "event_type": event_type,
                "event_time": event_time,

                "stress_before": stress_before,
                "stress_at_event": stress_at_event,
                "stress_after": stress_after,

                "baseline_at_event": baseline_at_event,

                "deviation_before": deviation_before,
                "deviation_at_event": deviation_at_event,
                "deviation_after": deviation_after,

                "high_pressure_threshold": threshold,

                "immediate_change": immediate_change,
                "after_change": after_change,

                "is_high_pressure": bool(
                    is_high_pressure
                ),
            }
        )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    result = pd.DataFrame(records)

    if result.empty:
        print()
        print("NO EVENT-STRESS MATCHES FOUND.")
        print()
        print(
            "This means the player/match identifiers or "
            "timestamps do not overlap."
        )
        return

    # --------------------------------------------------------
    # Recovery analysis
    #
    # Recovery is represented by the change from the
    # event stress level to the later stress level.
    # --------------------------------------------------------

    result["recovery_change"] = (
        result["stress_after"]
        - result["stress_at_event"]
    )

    result["has_recovery_window"] = (
        result["stress_after"].notna()
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "player_folder",
            "match_id",
            "event_time",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HIGH-PRESSURE EVENT RESULTS")
    print("=" * 70)

    print(f"Matched events: {len(result)}")

    print(
        f"Players: "
        f"{result['player_folder'].nunique()}"
    )

    print(
        f"Matches: "
        f"{result['match_id'].nunique()}"
    )

    print()
    print("Event types:")

    print(
        result["event_type"]
        .value_counts()
        .to_string()
    )

    print()
    print("High-pressure classification:")

    print(
        result["is_high_pressure"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "High-pressure percentage: "
        f"{100 * result['is_high_pressure'].mean():.2f}%"
    )

    print()
    print("Mean stress response by event type:")

    summary = (
        result
        .groupby("event_type")
        [
            [
                "stress_before",
                "stress_at_event",
                "stress_after",
                "immediate_change",
                "after_change",
            ]
        ]
        .mean()
        .round(4)
    )

    print(summary.to_string())

    print()
    print("High-pressure events by event type:")

    high_pressure_summary = (
        result[result["is_high_pressure"]]
        .groupby("event_type")
        .size()
    )

    print(
        high_pressure_summary.to_string()
    )

    print()
    print(f"Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()