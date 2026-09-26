import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="eSports Stress Detection",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .problem-box {
        padding: 25px;
        border-radius: 14px;
        border: 1px solid #374151;
        background: #111827;
        margin-bottom: 20px;
    }

    .solution-box {
        padding: 25px;
        border-radius: 14px;
        border: 1px solid #1d4ed8;
        background: #0f1d35;
        margin-bottom: 20px;
    }

    .pipeline-box {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #374151;
        background: #111827;
        text-align: center;
        margin: 10px 0 25px 0;
    }

    .pipeline-step {
        display: inline-block;
        padding: 10px 15px;
        margin: 5px;
        border-radius: 8px;
        background: #1f2937;
        border: 1px solid #4b5563;
        font-weight: 600;
    }

    .arrow {
        font-size: 22px;
        color: #60a5fa;
        margin: 0 3px;
    }

    .state-card {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #374151;
        background: #111827;
        text-align: center;
        min-height: 130px;
    }

    .state-label {
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 8px;
    }

    .state-value {
        font-size: 30px;
        font-weight: 750;
    }

    .future-box {
        padding: 24px;
        border-radius: 14px;
        background: #111827;
        border: 1px solid #374151;
    }

    .small-note {
        color: #9ca3af;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    predictions = pd.read_csv(
        REPORTS / "stress_predictions.csv"
    )

    events = pd.read_csv(
        REPORTS / "high_pressure_events.csv"
    )

    evaluation = pd.read_csv(
        REPORTS / "high_pressure_evaluation.csv"
    )

    model_comparison = pd.read_csv(
        REPORTS / "model_comparison.csv"
    )

    return (
        predictions,
        events,
        evaluation,
        model_comparison,
    )


try:

    (
        predictions,
        events,
        evaluation,
        model_comparison,
    ) = load_data()

except Exception as e:

    st.error(
        "Could not load the project reports."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# MODEL SELECTION
# ============================================================

selected_model = "Extra Trees"

selection_file = (
    REPORTS / "model_selection.json"
)

if selection_file.exists():

    try:

        with open(selection_file, "r") as f:

            selection = json.load(f)

        selected_model = selection.get(
            "selected_model",
            "Extra Trees"
        )

    except Exception:

        selected_model = "Extra Trees"


# ============================================================
# NORMALIZE TYPES
# ============================================================

events["is_high_pressure"] = (
    events["is_high_pressure"]
    .astype(bool)
)

predictions["window_start"] = pd.to_numeric(
    predictions["window_start"],
    errors="coerce"
)

predictions["window_end"] = pd.to_numeric(
    predictions["window_end"],
    errors="coerce"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎮 Project Controls")

players = sorted(
    predictions["player_folder"]
    .dropna()
    .unique()
)

selected_player = st.sidebar.selectbox(
    "Player",
    ["All Players"] + players
)

event_types = sorted(
    events["event_type"]
    .dropna()
    .unique()
)

selected_event = st.sidebar.selectbox(
    "Gameplay Event",
    ["All Events"] + list(event_types)
)

st.sidebar.divider()

st.sidebar.markdown(
    """
### Current Prototype

**Input:** Historical eSports sensor data

**Inference:** Replay-based

**ML:** Extra Trees

**Validation:** Leave-One-Player-Out
"""
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_predictions = predictions.copy()
filtered_events = events.copy()

if selected_player != "All Players":

    filtered_predictions = filtered_predictions[
        filtered_predictions["player_folder"]
        == selected_player
    ]

    filtered_events = filtered_events[
        filtered_events["player_folder"]
        == selected_player
    ]


if selected_event != "All Events":

    filtered_events = filtered_events[
        filtered_events["event_type"]
        == selected_event
    ]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎮 eSports Stress Detection & High-Pressure Event Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Multimodal machine learning for player-state estimation and gameplay-event analysis</div>',
    unsafe_allow_html=True,
)


# ============================================================
# REAL-WORLD PROBLEM
# ============================================================

st.markdown(
    '<div class="section-title">🌍 Real-World Problem</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="problem-box">

    <b>Competitive gaming generates important gameplay events such as
    kills, deaths and assists, but conventional game statistics mainly
    describe what happened in the game.</b>

    <br><br>

    They do not directly describe the player's physiological response
    during those moments.

    <br><br>

    The practical problem is therefore:

    <b>
    Can multimodal physiological and behavioral signals be used to
    estimate a player's stress-related state and identify gameplay
    events that occur during elevated estimated stress?
    </b>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OUR SOLUTION
# ============================================================

st.markdown(
    '<div class="section-title">💡 Our Solution</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="solution-box">

    The system combines <b>physiological signals</b>,
    <b>behavioral signals</b> and <b>timestamped gameplay events</b>.

    <br><br>

    A machine-learning model estimates a stress-related value from
    multimodal sensor features. A player-specific baseline is then
    used to identify periods of elevated estimated stress.

    <br><br>

    These periods are aligned with actual gameplay events to analyze
    <b>high-pressure-associated events and subsequent recovery</b>.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ How the System Works</div>',
    unsafe_allow_html=True,
)

pipeline = [
    "Sensor Data",
    "30-sec Windows",
    "Feature Engineering",
    "ML Model",
    "Predicted Stress",
    "Player Baseline",
    "Gameplay Event",
    "Pressure Analysis",
    "Recovery",
]

pipeline_html = (
    '<div class="pipeline-box">'
)

for i, step in enumerate(pipeline):

    pipeline_html += (
        f'<span class="pipeline-step">{step}</span>'
    )

    if i < len(pipeline) - 1:

        pipeline_html += (
            '<span class="arrow">→</span>'
        )

pipeline_html += "</div>"

st.markdown(
    pipeline_html,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📊 What Was Analyzed?</div>',
    unsafe_allow_html=True,
)

total_windows = len(filtered_predictions)

total_events = len(filtered_events)

high_pressure = int(
    filtered_events[
        "is_high_pressure"
    ].sum()
)

pressure_rate = (
    100 * high_pressure / total_events
    if total_events > 0
    else 0
)

matches = (
    filtered_predictions["match_id"]
    .nunique()
)

player_count = (
    filtered_predictions["player_folder"]
    .nunique()
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Sensor Windows",
        f"{total_windows:,}"
    )

with c2:
    st.metric(
        "Players",
        f"{player_count}"
    )

with c3:
    st.metric(
        "Matches",
        f"{matches}"
    )

with c4:
    st.metric(
        "Matched Events",
        f"{total_events:,}"
    )

with c5:
    st.metric(
        "Elevated-Stress Events",
        f"{high_pressure:,}"
    )


st.caption(
    f"{pressure_rate:.1f}% of the currently matched gameplay events "
    "met the player-relative elevated-stress criterion."
)

# ============================================================
# HIGH-PRESSURE EVENT EXPLORER
# ============================================================

st.markdown(
    '<div class="section-title">🔎 High-Pressure Event Explorer</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    Select an actual gameplay event from the dataset. The system
    compares model-estimated stress before, during, and after the
    event relative to the player's recent baseline.
    """
)

if filtered_events.empty:

    st.info(
        "No gameplay events are available for the current filters."
    )

else:

    # --------------------------------------------------------
    # Event selector
    # --------------------------------------------------------

    explorer_events = filtered_events.copy()

    explorer_events = explorer_events.sort_values(
        [
            "match_id",
            "event_time"
        ]
    )

    event_indices = explorer_events.index.tolist()

    selected_event_index = st.selectbox(
        "Choose a real gameplay event",
        event_indices,
        format_func=lambda idx: (
            f"{explorer_events.loc[idx, 'match_id']}  |  "
            f"{str(explorer_events.loc[idx, 'event_type']).upper()}  |  "
            f"{explorer_events.loc[idx, 'event_time']:.1f}s"
        ),
        key="event_explorer",
    )

    selected_row = explorer_events.loc[
        selected_event_index
    ]

    # --------------------------------------------------------
    # Event information
    # --------------------------------------------------------

    event_type = str(
        selected_row["event_type"]
    ).upper()

    event_time = float(
        selected_row["event_time"]
    )

    stress_before = selected_row[
        "stress_before"
    ]

    stress_event = selected_row[
        "stress_at_event"
    ]

    stress_after = selected_row[
        "stress_after"
    ]

    baseline = selected_row[
        "baseline_at_event"
    ]

    deviation = selected_row[
        "deviation_at_event"
    ]

    immediate_change = selected_row[
        "immediate_change"
    ]

    recovery_change = selected_row[
        "recovery_change"
    ]

    is_pressure = bool(
        selected_row["is_high_pressure"]
    )

    # --------------------------------------------------------
    # Event summary cards
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Gameplay Event",
            event_type
        )

    with c2:

        st.metric(
            "Stress Before",
            f"{stress_before:.4f}"
        )

    with c3:

        st.metric(
            "Stress At Event",
            f"{stress_event:.4f}",
            delta=f"{immediate_change:+.4f}"
        )

    with c4:

        st.metric(
            "Stress After",
            f"{stress_after:.4f}"
        )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if is_pressure:

        st.success(
            f"""
            **Elevated-stress-associated event**

            At {event_time:.1f} seconds, the model-estimated
            stress was {stress_event:.4f}, with a player-relative
            deviation of {deviation:+.4f}.

            The event therefore met the project's
            elevated-stress criterion.
            """
        )

    else:

        st.info(
            f"""
            **No elevated-stress association**

            At {event_time:.1f} seconds, the model-estimated
            stress was {stress_event:.4f}.

            This event did not meet the project's
            elevated-stress criterion.
            """
        )

    # --------------------------------------------------------
    # Baseline / response information
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Player Baseline",
            f"{baseline:.4f}"
            if pd.notna(baseline)
            else "N/A"
        )

    with c2:

        st.metric(
            "Event Deviation",
            f"{deviation:+.4f}"
            if pd.notna(deviation)
            else "N/A"
        )

    with c3:

        st.metric(
            "Recovery Change",
            f"{recovery_change:+.4f}"
            if pd.notna(recovery_change)
            else "N/A"
        )

    st.caption(
        "Association does not establish that the gameplay event "
        "caused the stress response."
    )

# ============================================================
# CURRENT PLAYER STATE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Current Estimated Player State</div>',
    unsafe_allow_html=True,
)

if filtered_predictions.empty:

    st.warning(
        "No prediction data available for the selected filter."
    )

else:

    latest = (
        filtered_predictions
        .sort_values(
            ["match_id", "window_start"]
        )
        .iloc[-1]
    )

    predicted = latest["predicted_stress"]
    baseline = latest["stress_baseline"]
    deviation = latest["stress_deviation"]
    state = latest["stress_state"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="state-card">
                <div class="state-label">Predicted Stress</div>
                <div class="state-value">{predicted:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        if pd.notna(baseline):

            baseline_text = f"{baseline:.4f}"

        else:

            baseline_text = "Insufficient"

        st.markdown(
            f"""
            <div class="state-card">
                <div class="state-label">Player Baseline</div>
                <div class="state-value">{baseline_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        if pd.notna(deviation):

            deviation_text = (
                f"{deviation:+.4f}"
            )

        else:

            deviation_text = "N/A"

        st.markdown(
            f"""
            <div class="state-card">
                <div class="state-label">Stress Deviation</div>
                <div class="state-value">{deviation_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="state-card">
                <div class="state-label">Estimated State</div>
                <div class="state-value">{state}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# STRESS + GAMEPLAY TIMELINE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Stress + Gameplay Timeline</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    The timeline combines sequential model-estimated stress with
    timestamped gameplay events. This allows us to examine whether
    elevated estimated stress occurs around important gameplay moments.
    """
)

if filtered_predictions.empty:

    st.warning(
        "No prediction data available."
    )

else:

    # --------------------------------------------------------
    # Select match for timeline
    # --------------------------------------------------------

    available_matches = sorted(
        filtered_predictions[
            "match_id"
        ]
        .dropna()
        .unique()
    )

    selected_match = st.selectbox(
        "Select Match",
        available_matches,
        key="timeline_match",
    )

    timeline_predictions = (
        filtered_predictions[
            filtered_predictions["match_id"]
            == selected_match
        ]
        .copy()
        .sort_values("window_start")
    )

    timeline_events = (
        filtered_events[
            filtered_events["match_id"]
            == selected_match
        ]
        .copy()
        .sort_values("event_time")
    )

    # --------------------------------------------------------
    # Stress chart
    # --------------------------------------------------------

    stress_chart_data = timeline_predictions[
        [
            "window_start",
            "predicted_stress",
            "stress_baseline",
        ]
    ].copy()

    stress_chart_data = (
        stress_chart_data
        .melt(
            id_vars=["window_start"],
            value_vars=[
                "predicted_stress",
                "stress_baseline",
            ],
            var_name="series",
            value_name="stress",
        )
        .dropna(subset=["stress"])
    )

    stress_chart_data["series"] = (
        stress_chart_data["series"]
        .replace(
            {
                "predicted_stress":
                    "Predicted Stress",
                "stress_baseline":
                    "Player Baseline",
            }
        )
    )

    stress_line = (
        alt.Chart(stress_chart_data)
        .mark_line(
            strokeWidth=3
        )
        .encode(
            x=alt.X(
                "window_start:Q",
                title="Time from Match Start (seconds)"
            ),
            y=alt.Y(
                "stress:Q",
                title="Stress Estimate"
            ),
            color=alt.Color(
                "series:N",
                title="Signal"
            ),
            tooltip=[
                alt.Tooltip(
                    "window_start:Q",
                    title="Time"
                ),
                alt.Tooltip(
                    "stress:Q",
                    title="Stress",
                    format=".4f"
                ),
                alt.Tooltip(
                    "series:N",
                    title="Signal"
                ),
            ],
        )
    )

    # --------------------------------------------------------
    # Gameplay event markers
    # --------------------------------------------------------

    if not timeline_events.empty:

        event_chart_data = timeline_events[
            [
                "event_time",
                "event_type",
                "is_high_pressure",
            ]
        ].copy()

        # Put event markers near the top of the chart.
        event_chart_data["marker_level"] = (
            timeline_predictions[
                "predicted_stress"
            ].max()
            * 1.02
        )

        event_points = (
            alt.Chart(event_chart_data)
            .mark_point(
                size=100,
                filled=True,
            )
            .encode(
                x=alt.X(
                    "event_time:Q",
                    title="Time from Match Start (seconds)"
                ),
                y=alt.Y(
                    "marker_level:Q",
                    title="Stress Estimate"
                ),
                shape=alt.Shape(
                    "event_type:N",
                    title="Gameplay Event"
                ),
                color=alt.Color(
                    "is_high_pressure:N",
                    title="Elevated Stress",
                ),
                tooltip=[
                    alt.Tooltip(
                        "event_time:Q",
                        title="Event Time",
                        format=".1f"
                    ),
                    alt.Tooltip(
                        "event_type:N",
                        title="Event"
                    ),
                    alt.Tooltip(
                        "is_high_pressure:N",
                        title="Elevated Stress"
                    ),
                ],
            )
        )

        combined_chart = (
            (stress_line + event_points)
            .resolve_scale(
                y="shared"
            )
            .properties(
                height=450
            )
            .interactive()
        )

    else:

        combined_chart = (
            stress_line
            .properties(
                height=450
            )
            .interactive()
        )

    st.altair_chart(
        combined_chart,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Event timeline table
    # --------------------------------------------------------

    if not timeline_events.empty:

        st.subheader(
            "Gameplay Events in This Match"
        )

        event_table = timeline_events[
            [
                "event_time",
                "event_type",
                "is_high_pressure",
            ]
        ].copy()

        event_table = event_table.rename(
            columns={
                "event_time":
                    "Time (seconds)",
                "event_type":
                    "Event",
                "is_high_pressure":
                    "Elevated-Stress Association",
            }
        )

        st.dataframe(
            event_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ML DECISION PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 From Sensor Data to Stress Decision</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    This is the core machine-learning stage of the system.
    Multimodal sensor measurements are converted into a
    30-second feature window and passed to the selected model.
    The model produces an estimated stress value, which is then
    interpreted relative to the player's recent baseline.
    """
)

# ------------------------------------------------------------
# Pipeline cards
# ------------------------------------------------------------

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.markdown(
        """
        <div class="state-card">
            <div class="state-label">INPUT</div>
            <div class="state-value">Multimodal<br>Signals</div>
            <br>
            <span class="small-note">
            HR • GSR • EMG • Eye • SpO2<br>
            Movement • Keyboard • Mouse
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:

    st.markdown(
        """
        <div class="state-card">
            <div class="state-label">FEATURE WINDOW</div>
            <div class="state-value">30 sec</div>
            <br>
            <span class="small-note">
            Statistical and temporal features
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:

    st.markdown(
        f"""
        <div class="state-card">
            <div class="state-label">ML MODEL</div>
            <div class="state-value">{selected_model}</div>
            <br>
            <span class="small-note">
            Leave-One-Player-Out validation
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p4:

    if not filtered_predictions.empty:

        latest_ml = (
            filtered_predictions
            .sort_values(
                ["match_id", "window_start"]
            )
            .iloc[-1]
        )

        prediction_value = (
            latest_ml["predicted_stress"]
        )

        ml_state = (
            latest_ml["stress_state"]
        )

    else:

        prediction_value = np.nan
        ml_state = "N/A"

    prediction_text = (
        f"{prediction_value:.4f}"
        if pd.notna(prediction_value)
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="state-card">
            <div class="state-label">MODEL OUTPUT</div>
            <div class="state-value">{prediction_text}</div>
            <br>
            <span class="small-note">
            Estimated stress
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Decision stage
# ------------------------------------------------------------

st.markdown(
    """
    <div class="pipeline-box">

        <span class="pipeline-step">
            Predicted Stress
        </span>

        <span class="arrow">−</span>

        <span class="pipeline-step">
            Player Baseline
        </span>

        <span class="arrow">=</span>

        <span class="pipeline-step">
            Stress Deviation
        </span>

        <span class="arrow">→</span>

        <span class="pipeline-step">
            Player-Relative State
        </span>

    </div>
    """,
    unsafe_allow_html=True,
)


if not filtered_predictions.empty:

    latest_ml = (
        filtered_predictions
        .sort_values(
            ["match_id", "window_start"]
        )
        .iloc[-1]
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Predicted Stress",
            f"{latest_ml['predicted_stress']:.4f}"
        )

    with c2:

        if pd.notna(
            latest_ml["stress_baseline"]
        ):

            st.metric(
                "Recent Player Baseline",
                f"{latest_ml['stress_baseline']:.4f}"
            )

        else:

            st.metric(
                "Recent Player Baseline",
                "Insufficient"
            )

    with c3:

        if pd.notna(
            latest_ml["stress_deviation"]
        ):

            st.metric(
                "Stress Deviation",
                f"{latest_ml['stress_deviation']:+.4f}"
            )

        else:

            st.metric(
                "Stress Deviation",
                "N/A"
            )

    st.info(
        f"""
        **Current model-estimated state: {latest_ml['stress_state']}**

        The state is determined relative to the player's recent
        predicted-stress baseline rather than using a fixed
        heart-rate threshold.
        """
    )
    with st.expander("🔬 What goes into one 30-second feature window?"):

            st.markdown(
            """
            ### Physiological signals
    
            - ❤️ Heart rate
            - 〰️ Galvanic skin response (GSR)
            - 💪 EMG — right and left hand
            - 🌡 Facial skin temperature
            - 🫁 SpO2
    
            ### Behavioral / interaction signals
    
            - 👁 Eye movement
            - 👁 Pupil diameter
            - 🧠 Head movement
            - ⌨ Keyboard activity
            - 🖱 Mouse movement
            - 🖱 Mouse clicks
    
            ### Feature engineering
    
            For each 30-second window, statistical and temporal
            characteristics such as mean, standard deviation,
            minimum, maximum, median, range, percentiles and
            temporal trend are calculated.
    
            These engineered features form the model input.
            """
        )
# ============================================================
# EVENT EVIDENCE
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Gameplay Event Evidence</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    The system does not label every gameplay event as stressful.
    It checks whether the event occurs during a period of elevated
    model-estimated stress relative to the player's baseline.
    """
)

if not filtered_events.empty:

    event_counts = (
        filtered_events["event_type"]
        .value_counts()
        .rename("Events")
    )

    high_counts = (
        filtered_events[
            filtered_events["is_high_pressure"]
        ]["event_type"]
        .value_counts()
        .rename("Elevated-Stress Events")
    )

    event_summary = pd.concat(
        [
            event_counts,
            high_counts,
        ],
        axis=1,
    ).fillna(0)

    event_summary[
        "Elevated-Stress Rate (%)"
    ] = (
        100
        * event_summary[
            "Elevated-Stress Events"
        ]
        / event_summary["Events"]
    )

    event_summary = event_summary.round(2)

    st.dataframe(
        event_summary,
        use_container_width=True,
    )


# ============================================================
# EXAMPLE EVENT
# ============================================================

st.markdown(
    '<div class="section-title">🔎 What Does One Event Look Like?</div>',
    unsafe_allow_html=True,
)

if not filtered_events.empty:

    example_options = filtered_events.index.tolist()

    example_index = st.selectbox(
        "Choose an event to inspect",
        example_options,
        format_func=lambda x:
            (
                f"{filtered_events.loc[x, 'match_id']} | "
                f"{filtered_events.loc[x, 'event_type']} | "
                f"{filtered_events.loc[x, 'event_time']:.1f}s"
            ),
    )

    example = filtered_events.loc[
        example_index
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Event",
            str(example["event_type"]).upper()
        )

    with c2:

        st.metric(
            "Stress Before",
            f"{example['stress_before']:.4f}"
        )

    with c3:

        st.metric(
            "Stress At Event",
            f"{example['stress_at_event']:.4f}"
        )

    with c4:

        st.metric(
            "Stress After",
            f"{example['stress_after']:.4f}"
        )

    if example["is_high_pressure"]:

        st.success(
            "This event meets the project's "
            "elevated-stress-associated criterion."
        )

    else:

        st.info(
            "This event does not meet the project's "
            "elevated-stress-associated criterion."
        )

    st.caption(
        "Association with elevated estimated stress does not "
        "establish that the gameplay event caused the stress response."
    )


# ============================================================
# MODEL EVIDENCE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Why This Machine-Learning Model?</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    Rather than assuming Random Forest was the correct algorithm,
    multiple regression models were evaluated using
    Leave-One-Player-Out validation.
    """
)

model_display = model_comparison.copy()

st.dataframe(
    model_display,
    use_container_width=True,
    hide_index=True,
)

selected_rows = model_comparison[
    model_comparison["model"]
    == selected_model
]

if not selected_rows.empty:

    selected_row = selected_rows.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Selected Model",
            selected_model
        )

    with c2:

        st.metric(
            "MAE",
            f"{selected_row['MAE_mean']:.5f}"
        )

    with c3:

        st.metric(
            "RMSE",
            f"{selected_row['RMSE_mean']:.5f}"
        )

    st.warning(
        f"""
        R² = {selected_row['R2_mean']:.5f}.
        The negative R² indicates limited cross-player
        generalization. Therefore, the current system should
        be presented as a research prototype rather than as a
        highly accurate universal stress predictor.
        """
    )


# ============================================================
# PRACTICAL APPLICATION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Real-World Use of the System</div>',
    unsafe_allow_html=True,
)

use1, use2, use3 = st.columns(3)

with use1:

    st.markdown(
        """
        ### 🎮 Player Performance

        Identify gameplay periods where elevated
        stress-related responses repeatedly occur.
        """
    )

with use2:

    st.markdown(
        """
        ### 🧑‍🏫 Coaching Analysis

        Review gameplay events alongside estimated
        physiological responses instead of relying only
        on conventional game statistics.
        """
    )

with use3:

    st.markdown(
        """
        ### 📊 Performance Research

        Compare event timing, estimated stress elevation
        and recovery across players and sessions.
        """
    )


# ============================================================
# CURRENT VS FUTURE SYSTEM
# ============================================================

st.markdown(
    '<div class="section-title">🚀 Current Prototype → Future System</div>',
    unsafe_allow_html=True,
)

current, future = st.columns(2)

with current:

    st.markdown(
        """
        <div class="future-box">

        <h3>Current Prototype</h3>

        Historical eSports sensor data

        ↓

        30-second feature windows

        ↓

        Machine-learning inference

        ↓

        Gameplay-event analysis

        <br>

        <b>Replay-based sequential analysis</b>

        </div>
        """,
        unsafe_allow_html=True,
    )

with future:

    st.markdown(
        """
        <div class="future-box">

        <h3>Future Deployment</h3>

        Live wearable sensors

        +

        Live gameplay telemetry

        ↓

        Streaming feature extraction

        ↓

        Real-time stress estimation

        ↓

        Player/coaching feedback

        <br>

        <b>Live sequential inference</b>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RESEARCH LIMITATIONS
# ============================================================

with st.expander(
    "📚 Research Methodology & Limitations"
):

    st.markdown(
        """
        ### Dataset

        - 4 players in the current multimodal subset
        - 15 matches
        - 30-second sensor windows
        - 1,548 prediction windows

        ### Target

        The target is an EEG-derived stress metric supplied by
        the source dataset. It is treated as a stress-related
        proxy, not clinical ground truth.

        ### Validation

        Leave-One-Player-Out validation was used to evaluate
        generalization to unseen players.

        ### High-Pressure Definition

        A gameplay event is considered elevated-stress-associated
        when it occurs during a period where model-estimated
        stress deviation is elevated relative to the player's
        baseline.

        ### Limitations

        - Cross-player generalization remains limited.
        - The stress target is a dataset-derived proxy.
        - Event association does not establish causality.
        - The current implementation is replay-based.
        - More players and matches are required for stronger
          generalization.
        - This system is not a medical diagnostic tool.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "eSports Stress Detection & High-Pressure Event Analysis | "
    "Research Prototype"
)
