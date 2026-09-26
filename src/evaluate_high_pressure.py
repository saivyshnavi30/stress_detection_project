from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "reports" / "high_pressure_events.csv"
OUTPUT_FILE = ROOT / "reports" / "high_pressure_evaluation.csv"


def main():

    print("=" * 70)
    print("HIGH-PRESSURE EVENT EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"Events: {len(df)}")

    # --------------------------------------------------------
    # Basic groups
    # --------------------------------------------------------

    df["is_high_pressure"] = (
        df["is_high_pressure"].astype(bool)
    )

    # --------------------------------------------------------
    # Overall comparison
    # --------------------------------------------------------

    overall = (
        df.groupby("is_high_pressure")
        [
            [
                "stress_before",
                "stress_at_event",
                "stress_after",
                "immediate_change",
                "after_change",
                "deviation_at_event",
            ]
        ]
        .agg(["mean", "median", "std"])
    )

    print()
    print("=" * 70)
    print("HIGH-PRESSURE VS OTHER EVENTS")
    print("=" * 70)

    print(
        overall.round(5).to_string()
    )

    # --------------------------------------------------------
    # Event-type comparison
    # --------------------------------------------------------

    by_event = (
        df.groupby(
            [
                "event_type",
                "is_high_pressure"
            ]
        )
        [
            [
                "stress_before",
                "stress_at_event",
                "stress_after",
                "immediate_change",
                "after_change",
                "deviation_at_event",
            ]
        ]
        .agg(["mean", "median", "count"])
    )

    print()
    print("=" * 70)
    print("BY EVENT TYPE")
    print("=" * 70)

    print(
        by_event.round(5).to_string()
    )

    # --------------------------------------------------------
    # High-pressure proportion by event type
    # --------------------------------------------------------

    proportions = (
        df.groupby("event_type")["is_high_pressure"]
        .agg(
            total_events="count",
            high_pressure_events="sum",
            high_pressure_rate="mean",
        )
        .reset_index()
    )

    proportions["high_pressure_rate"] *= 100

    print()
    print("=" * 70)
    print("HIGH-PRESSURE RATE BY EVENT TYPE")
    print("=" * 70)

    print(
        proportions.round(2).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Player-level analysis
    # --------------------------------------------------------

    player_summary = (
        df.groupby("player_folder")
        .agg(
            events=("event_type", "count"),
            high_pressure_events=(
                "is_high_pressure",
                "sum"
            ),
            mean_stress_at_event=(
                "stress_at_event",
                "mean"
            ),
            mean_immediate_change=(
                "immediate_change",
                "mean"
            ),
            mean_recovery_change=(
                "recovery_change",
                "mean"
            ),
        )
        .reset_index()
    )

    player_summary["high_pressure_rate"] = (
        100
        * player_summary["high_pressure_events"]
        / player_summary["events"]
    )

    print()
    print("=" * 70)
    print("PLAYER-LEVEL SUMMARY")
    print("=" * 70)

    print(
        player_summary.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save compact evaluation table
    # --------------------------------------------------------

    evaluation_rows = []

    for event_type in sorted(
        df["event_type"].unique()
    ):

        subset = df[
            df["event_type"] == event_type
        ]

        for high_pressure in [False, True]:

            group = subset[
                subset["is_high_pressure"]
                == high_pressure
            ]

            if group.empty:
                continue

            evaluation_rows.append(
                {
                    "event_type": event_type,
                    "high_pressure": high_pressure,
                    "n": len(group),
                    "mean_stress_before":
                        group["stress_before"].mean(),
                    "mean_stress_at_event":
                        group["stress_at_event"].mean(),
                    "mean_stress_after":
                        group["stress_after"].mean(),
                    "mean_immediate_change":
                        group["immediate_change"].mean(),
                    "mean_after_change":
                        group["after_change"].mean(),
                    "mean_deviation_at_event":
                        group["deviation_at_event"].mean(),
                }
            )

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    evaluation.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()