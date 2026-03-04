"""
Temporal Fluency Analyzer – Streamlit App
Analyzes speech rate, articulation rate, pause behavior, MLR,
and phonation ratio from audio recordings using Praat
(via praat-parselmouth).
"""

import math
import os
import tempfile
import traceback
from datetime import datetime

import pandas as pd
import parselmouth
import streamlit as st
from parselmouth.praat import call

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Temporal Fluency Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "aiff", "aifc", "au", "flac"]

METRIC_HELP = {
    "Duration (s)":               "Total length of the audio recording.",
    "Syllables":                  "Number of voiced syllable peaks detected.",
    "Pauses":                     "Number of silent pauses above the minimum duration.",
    "Phonation Time (s)":         "Total time spent producing voiced speech.",
    "Speech Rate (syll/s)":       "Syllables ÷ total duration.",
    "Articulation Rate (syll/s)": "Syllables ÷ phonation time (pauses excluded).",
    "Mean Syllable Duration (s)": "Phonation time ÷ syllable count.",
    "MLR":                        "Mean Length of Run – syllables ÷ pause count.",
    "Phonation Ratio":            "Phonation time ÷ total duration.",
}


# ── Core analysis ──────────────────────────────────────────────────────────────
def analyze_audio(
    file_path:  str,
    assignment: str   = "Unknown",
    silence_db: float = -25.0,
    min_dip:    float = 2.0,
    min_pause:  float = 0.3,
    display_name: str   = None,       # ← add this
) -> dict:
    """
    Praat-based temporal fluency analysis.

    Parameters
    ----------
    file_path   : Path to a Praat-readable audio file.
    assignment  : Label for the recording session / assignment.
    silence_db  : Intensity threshold (dB) below the 99th-percentile maximum
                  used to detect silence.
    min_dip     : Minimum intensity dip (dB) required between syllable peaks.
    min_pause   : Minimum silence duration (s) counted as a pause.

    Returns
    -------
    dict of fluency measures.
    """
    sound       = parselmouth.Sound(file_path)
    originaldur = sound.get_total_duration()

    # ── Intensity envelope ─────────────────────────────────────────────────
    intensity     = sound.to_intensity(50)
    min_intensity = call(intensity, "Get minimum",  0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum",  0, 0, "Parabolic")
    max_99        = call(intensity, "Get quantile", 0, 0, 0.99)

    threshold  = max_99 + silence_db
    threshold3 = silence_db - (max_intensity - max_99)
    if threshold < min_intensity:
        threshold = min_intensity

    # ── Silence / phonation time ───────────────────────────────────────────
    textgrid     = call(intensity, "To TextGrid (silences)",
                        threshold3, min_pause, 0.1, "silent", "sounding")
    silencetier  = call(textgrid,  "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses      = call(silencetable, "Get number of rows")

    speakingtot = 0.0
    for i in range(npauses):
        row          = i + 1
        speakingtot += (call(silencetable, "Get value", row, 2) -
                        call(silencetable, "Get value", row, 1))

    # ── Syllable peak detection ────────────────────────────────────────────
    int_matrix    = call(intensity, "Down to Matrix")
    int_sound     = call(int_matrix, "To Sound (slice)", 1)
    int_duration  = call(int_sound,  "Get total duration")
    point_process = call(int_sound,  "To PointProcess (extrema)",
                         "Left", "yes", "no", "Sinc70")
    numpeaks      = call(point_process, "Get number of points")
    t_all         = [call(point_process, "Get time from index", i + 1)
                     for i in range(numpeaks)]

    timepeaks, intensities = [], []
    for ti in t_all:
        v = call(int_sound, "Get value at time", ti, "Cubic")
        if v > threshold:
            timepeaks.append(ti)
            intensities.append(v)

    if not timepeaks:
        raise ValueError(
            "No usable intensity peaks found – "
            "the recording may be silent or too short."
        )

    # Keep peaks where the preceding dip exceeds min_dip
    validtime   = []
    currenttime = timepeaks[0]
    currentint  = intensities[0]
    for p in range(len(timepeaks) - 1):
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        if abs(currentint - dip) > min_dip:
            validtime.append(timepeaks[p])
        currenttime = timepeaks[p + 1]
        currentint  = call(int_sound, "Get value at time", timepeaks[p + 1], "Cubic")

    # ── Voiced peak selection ──────────────────────────────────────────────
    pitch      = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25,
                                   0.01, 0.35, 0.25, 450)
    voicedpeak = []
    for qtime in validtime:
        interval = call(textgrid, "Get interval at time", 1, qtime)
        label    = call(textgrid, "Get label of interval", 1, interval)
        if label == "sounding" and not math.isnan(pitch.get_value_at_time(qtime)):
            voicedpeak.append(qtime)

    voicedcount = len(voicedpeak)
    if voicedcount == 0:
        raise ValueError(
            "No voiced segments detected – please check the audio quality."
        )

    # Insert syllable marks (time-corrected)
    timecorr = originaldur / int_duration
    call(textgrid, "Insert point tier", 1, "syllables")
    for vp in voicedpeak:
        call(textgrid, "Insert point", 1, vp * timecorr, "")

    # ── Derived measures ───────────────────────────────────────────────────
    npause           = npauses - 1
    speakingrate     = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    asd              = speakingtot / voicedcount
    mlr              = voicedcount / npause if npause > 0 else float(voicedcount)
    phonation_ratio  = speakingtot / originaldur

    return {
        "Filename":                   os.path.basename(file_path),
        "Assignment":                 assignment,
        "Duration (s)":               round(originaldur, 2),
        "Syllables":                  voicedcount,
        "Pauses":                     npause,
        "Phonation Time (s)":         round(speakingtot, 3),
        "Speech Rate (syll/s)":       round(speakingrate, 2),
        "Articulation Rate (syll/s)": round(articulationrate, 2),
        "Mean Syllable Duration (s)": round(asd, 3),
        "MLR":                        round(mlr, 2),
        "Phonation Ratio":            round(phonation_ratio, 3),
    }


# ── File helpers ───────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    """
    Write an UploadedFile into a temp directory using its original filename.
    Non-WAV files are converted to WAV via pydub when ffmpeg is available.
    The caller is responsible for cleanup via cleanup().
    """
    tmp_dir   = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix != ".wav":
        try:
            from pydub import AudioSegment
            audio    = AudioSegment.from_file(file_path)
            wav_path = os.path.join(
                tmp_dir,
                os.path.splitext(uploaded_file.name)[0] + ".wav"
            )
            audio.export(wav_path, format="wav")
            os.unlink(file_path)
            return wav_path
        except Exception:
            pass

    return file_path


def cleanup(path: str):
    """Silently remove a temp file and its parent temp directory."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
        parent = os.path.dirname(path)
        if parent and os.path.isdir(parent):
            os.rmdir(parent)   # only succeeds if empty, which it will be
    except OSError:
        pass


def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar() -> tuple[float, float, float]:
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Analysis Parameters")
        silence_db = st.slider(
            "Silence threshold (dB)",
            min_value=-40, max_value=-10, value=-25, step=1,
            help="Intensity level below the 99th-percentile maximum treated as silence.",
        )
        min_dip = st.slider(
            "Minimum dip between peaks (dB)",
            min_value=1, max_value=10, value=2, step=1,
            help="Minimum intensity dip required to separate two syllable peaks.",
        )
        min_pause = st.slider(
            "Minimum pause duration (s)",
            min_value=0.05, max_value=1.0, value=0.3, step=0.05,
            help="Shortest silence counted as a pause.",
        )

        st.divider()
        st.subheader("ℹ️ Measure Glossary")
        for metric, description in METRIC_HELP.items():
            st.caption(f"**{metric}**: {description}")

    return float(silence_db), float(min_dip), float(min_pause)


# ── Shared result display ──────────────────────────────────────────────────────
def show_metrics(result: dict):
    """Three-column metric card layout."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏱ Duration",        f"{result['Duration (s)']} s")
        st.metric("🗣 Syllables",       result["Syllables"])
        st.metric("⏸ Pauses",          result["Pauses"])
    with col2:
        st.metric("📈 Speech Rate (Native = 3.3+)",        f"{result['Speech Rate (syll/s)']} syll/s")
        st.metric("⚡ Articulation Rate",  f"{result['Articulation Rate (syll/s)']} syll/s")
        st.metric("📏 MLR (Native = 7+)",                result["MLR"])
    with col3:
        st.metric("🎙 Phonation Time",  f"{result['Phonation Time (s)']} s")
        st.metric("📊 Phonation Ratio (higher is better)", f"{result['Phonation Ratio']:.1%}")
        st.metric("🔤 Mean Syll. Dur.", f"{result['Mean Syllable Duration (s)']} s")


# ── Single-file mode ───────────────────────────────────────────────────────────
def single_file_ui(silence_db: float, min_dip: float, min_pause: float):
    st.subheader("📄 Single File Analysis")

    col_upload, col_name = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload an audio file",
            type=SUPPORTED_FORMATS,
            help=f"Supported: {', '.join(SUPPORTED_FORMATS).upper()}",
        )
    with col_name:
        assignment = st.text_input("Assignment name", value="Assignment 1")

    if uploaded is None:
        st.info("⬆️ Upload an audio file to get started.")
        return

    st.audio(uploaded)

    if st.button("▶ Analyze", type="primary"):
        st.session_state.pop("single_result", None)
        tmp = None
        with st.spinner(f"Analyzing **{uploaded.name}** …"):
            try:
                tmp    = save_upload(uploaded)
                result = analyze_audio(tmp, assignment, silence_db, min_dip, min_pause,display_name=uploaded.name,)
                st.session_state["single_result"] = result
            except Exception as exc:
                st.error(f"❌ Analysis failed: {exc}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())
            finally:
                cleanup(tmp)

    if "single_result" in st.session_state:
        result = st.session_state["single_result"]
        st.success("✅ Analysis complete!")
        show_metrics(result)

        df = pd.DataFrame([result])
        with st.expander("📋 Full data table"):
            st.dataframe(df, use_container_width=True)

        stem = os.path.splitext(uploaded.name)[0]
        st.download_button(
            label     = "📥 Download CSV",
            data      = df_to_csv(df),
            file_name = f"fluency_{stem}_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime      = "text/csv",
        )


# ── Batch mode ─────────────────────────────────────────────────────────────────
def batch_ui(silence_db: float, min_dip: float, min_pause: float):
    st.subheader("📁 Batch Processing")

    col_upload, col_name = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload audio files",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(SUPPORTED_FORMATS).upper()}",
        )
    with col_name:
        assignment = st.text_input("Assignment name", value="Assignment 1")

    if not uploaded_files:
        st.info("⬆️ Upload one or more audio files to get started.")
        return

    n = len(uploaded_files)
    st.info(f"**{n}** file(s) queued.")

    if st.button("▶ Analyze All", type="primary"):
        st.session_state.pop("batch_df", None)
        results: list[dict] = []
        errors:  list[tuple] = []

        progress = st.progress(0.0, text="Starting…")
        status   = st.empty()

        for idx, uf in enumerate(uploaded_files):
            progress.progress(idx / n, text=f"Processing {idx + 1} / {n}: {uf.name}")
            status.caption(f"Current file: **{uf.name}**")
            tmp = None
            try:
                tmp = save_upload(uf)
                results.append(
                    analyze_audio(tmp, assignment, silence_db, min_dip, min_pause,display_name=uf.name,)
                )
            except Exception as exc:
                errors.append((uf.name, str(exc)))
            finally:
                cleanup(tmp)

        progress.progress(1.0, text="Done!")
        status.empty()

        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) failed"):
                for fname, msg in errors:
                    st.warning(f"**{fname}**: {msg}")

        if results:
            st.session_state["batch_df"] = pd.DataFrame(results)
            st.success(f"✅ {len(results)} / {n} files processed successfully.")

    # ── Display persisted results ──────────────────────────────────────────
    if "batch_df" not in st.session_state:
        return

    df = st.session_state["batch_df"]

    # Summary statistics
    st.subheader("📊 Summary Statistics")
    stat_cols = [
        "Duration (s)", "Speech Rate (syll/s)",
        "Articulation Rate (syll/s)", "MLR", "Phonation Ratio",
    ]
    st.dataframe(df[stat_cols].describe().round(3), use_container_width=True)

    # Charts
    st.subheader("📈 Charts")
    plot_df = df.set_index("Filename")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Speech Rate (syll/s)")
        st.bar_chart(plot_df["Speech Rate (syll/s)"])
    with c2:
        st.caption("MLR")
        st.bar_chart(plot_df["MLR"])
    with c3:
        st.caption("Phonation Ratio")
        st.bar_chart(plot_df["Phonation Ratio"])

    # Full results table
    st.subheader("📋 All Results")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label     = "📥 Download CSV",
        data      = df_to_csv(df),
        file_name = f"fluency_batch_{assignment}_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime      = "text/csv",
    )


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    silence_db, min_dip, min_pause = sidebar()

    st.title("🎙️ Temporal Fluency Analyzer")
    st.markdown(
        "Analyze **speech rate**, **articulation rate**, **pause behavior**, "
        "**MLR**, and **phonation ratio** from audio recordings."
    )
    st.divider()

    mode = st.radio(
        "Select mode:",
        ["📄 Single File", "📁 Batch Processing"],
        horizontal=True,
    )
    st.divider()

    if mode == "📄 Single File":
        single_file_ui(silence_db, min_dip, min_pause)
    else:
        batch_ui(silence_db, min_dip, min_pause)


if __name__ == "__main__":
    main()
