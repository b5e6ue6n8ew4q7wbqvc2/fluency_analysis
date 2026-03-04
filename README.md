# 🎙️ Temporal Fluency Analyzer

A Streamlit web app for analyzing temporal fluency measures in student audio
recordings using Praat (via [praat-parselmouth](https://parselmouth.readthedocs.io/)).

## Features

- **Single file** or **batch** processing modes
- Measures reported for every recording:
  | Measure | Description |
  |---|---|
  | Duration (s) | Total recording length |
  | Syllables | Voiced syllable peak count |
  | Pauses | Silent pause count |
  | Phonation Time (s) | Total voiced speech time |
  | Speech Rate (syll/s) | Syllables ÷ duration |
  | Articulation Rate (syll/s) | Syllables ÷ phonation time |
  | Mean Syllable Duration (s) | Phonation time ÷ syllables |
  | MLR | Syllables ÷ pauses |
  | Phonation Ratio | Phonation time ÷ duration |
- Adjustable analysis parameters via sidebar sliders
- Summary statistics and bar charts (batch mode)
- One-click CSV download

## Supported Audio Formats

`WAV` · `MP3` · `OGG` · `AIFF` · `AIFC` · `AU` · `FLAC`

> **Note:** Non-WAV formats require `ffmpeg`. WAV files work natively with Praat.

## Running Locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) Install ffmpeg for non-WAV format support
#    macOS:  brew install ffmpeg
#    Linux:  sudo apt install ffmpeg
#    Windows: https://ffmpeg.org/download.html

# 5. Launch
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. `packages.txt` installs `ffmpeg` automatically on the cloud runner — no extra steps needed.

## Analysis Parameters

Adjustable in the sidebar at runtime:

| Parameter | Default | Effect |
|---|---|---|
| Silence threshold | −25 dB | dB below the 99th-percentile max treated as silence |
| Min dip between peaks | 2 dB | Intensity dip required to separate syllable peaks |
| Min pause duration | 0.3 s | Shortest silence counted as a pause |

## Analysis Method

The pipeline follows the Praat syllable-nuclei detection approach from
De Jong & Wempe (2009), adapted here via `praat-parselmouth`:

1. Compute the intensity envelope
2. Detect sounding/silent intervals using the dB threshold
3. Find intensity peaks separated by sufficient dips (syllable candidates)
4. Retain only voiced peaks confirmed by pitch analysis
5. Calculate rate and pause measures from the resulting counts

## Project Structure

```
.
├── app.py            # Streamlit application
├── requirements.txt  # Python dependencies
├── packages.txt      # System packages for Streamlit Cloud (ffmpeg)
├── .gitignore
└── README.md
```
