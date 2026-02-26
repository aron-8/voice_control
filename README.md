# 🎙️ Voice Control

A lightweight keyword-spotting system for hands-free computer control. Speak a word, trigger a keypress. Runs fully offline on CPU.

Built with Python, TensorFlow/Keras, librosa, and sounddevice.

---

## How It Works

```
record.py  →  train.py  →  live_recognize.py
  (collect)     (learn)        (listen & act)
```

1. **Record** — say each word into your microphone, clips are saved automatically when voice activity is detected
2. **Train** — MFCC features are extracted from the clips, augmented 6×, and used to train a small CNN
3. **Recognize** — the trained model listens in real time and maps recognized words to keypresses

---

## Requirements

```
pip install sounddevice soundfile librosa numpy scikit-learn tensorflow joblib pyautogui
```

> On Linux you may also need: `sudo apt install libportaudio2`

---

## File Structure

```
voice_control/
├── config.py             ← all settings live here
├── record.py             ← step 1: collect audio samples
├── train.py              ← step 2: train the model
├── live_recognize.py     ← step 3: real-time recognition
│
├── data/                 ← created automatically by record.py
│   ├── forward/
│   ├── back/
│   └── noise/
│
├── word_recog_model.keras
├── label_encoder.pkl
└── norm_stats.npz
```

---

## Quick Start

### 1. Configure your words

Open `config.py` and edit the `WORDS` list:

```python
WORDS = [
    "forward",
    "back",
    "left",
    "right",
    "stop",
    "play",
    "pause",
    "volume",
    "noise",   # ← always keep this — background noise class
]
```

### 2. Record samples

```bash
python record.py
```

The script will loop through each word in `WORDS` and prompt you to speak. It uses voice-activation — recording triggers automatically when you speak. The `noise` class is recorded continuously without a trigger (just let it capture background sound).

**Tips for good recordings:**
- Speak naturally, at the volume you'll use during real use
- Vary your tone and speed slightly between samples
- Move slightly between recordings to capture microphone variation
- Record in the same environment where you'll use the system

### 3. Train the model

```bash
python train.py
```

Training runs for up to 50 epochs with early stopping. On CPU this takes 1–3 minutes depending on dataset size. Three files are saved when done:

| File | Purpose |
|---|---|
| `word_recog_model.keras` | the trained neural network |
| `label_encoder.pkl` | maps class indices to word names |
| `norm_stats.npz` | normalization stats (must match training) |

### 4. Run live recognition

```bash
python live_recognize.py
```

Stay silent for ~1 second during calibration, then start speaking. Press `Ctrl+C` to quit.

---

## Adding Custom Words

### Step 1 — Add to `config.py`

```python
WORDS = [
    "forward",
    "back",
    "stop",
    "play",
    "noise",
    "mute",      # ← new word added here
    "screenshot", # ← another new word
]
```

> **Word length limit:** `DURATION = 1` second. Words up to 2 syllables fit comfortably. For 3-syllable words increase `DURATION` to `1.5` in `config.py` — but you must re-record and retrain everything if you change this.

### Step 2 — Record the new word

```bash
python record.py
```

The script picks up where it left off — it skips words that already have enough samples and only records missing or new ones.

### Step 3 — Map the word to an action in `live_recognize.py`

Open `live_recognize.py` and find the `dispatch()` function:

```python
def dispatch(label: str):
    action_map = {
        'forward':    lambda: send_key('right'),
        'back':       lambda: send_key('left'),
        'stop':       lambda: send_key('space'),
        'play':       lambda: send_key('space'),
        'mute':       lambda: send_key('m'),       # ← add your word here
        'screenshot': lambda: send_key('s'),       # ← and map it to a key
    }
```

Available key names for `send_key()`: any single letter (`'a'`–`'z'`), `'space'`, `'left'`, `'right'`, `'up'`, `'down'`, or any [pyautogui key name](https://pyautogui.readthedocs.io/en/latest/keyboard.html).

### Step 4 — Retrain

```bash
python train.py
```

Always retrain from scratch when adding new words — the model must learn all classes together.

---

## Removing a Word

1. Remove it from `WORDS` in `config.py`
2. Delete its folder from `data/` (optional but keeps things clean)
3. Remove its entry from `dispatch()` in `live_recognize.py`
4. Retrain: `python train.py`

---

## Tuning

All tunable parameters are in `config.py`. The most useful ones:

| Parameter | Default | Effect |
|---|---|---|
| `SAMPLES_PER_WORD` | `50` | More = better accuracy, longer recording session |
| `THRESHOLD_MULTIPLIER` | `2.0` | Higher = less sensitive trigger (fewer false triggers) |
| `CONF_THRESHOLD` | `0.6` | Higher = only act on very confident predictions |
| `DURATION` | `1` | Seconds per clip — increase for longer words |
| `N_MFCC` | `13` | More coefficients = richer features, slower training |

> If you change `DURATION` or `N_MFCC` you must re-record **and** retrain everything.

---

## How Many Samples Do I Need?

| Samples per word | Expected accuracy |
|---|---|
| 5–10 | Pipeline test only — not reliable |
| 20–30 | Decent for quiet, consistent environments |
| 50+ | Recommended for real use |
| 100+ | Best for noisy environments or many similar-sounding words |

With augmentation enabled in `train.py`, each recording generates 6 training samples (1 original + 5 augmented), so 50 recordings → 300 effective training samples per word.

---

## Troubleshooting

**Trigger fires constantly / too easily**
Increase `THRESHOLD_MULTIPLIER` in `config.py` (try `3.0`) or re-run and stay quieter during calibration.

**Words not recognized / always predicts noise**
Check that `norm_stats.npz` exists — if it's missing, run `train.py` again. Make sure you haven't changed `DURATION`, `N_MFCC`, or `HOP_LENGTH` between training and recognition.

**Wrong word predicted**
Record more samples, especially in varied conditions. Pay attention to similar-sounding words — "play" and "stay" confuse models easily. Rename one if needed.

**Shape error on prediction**
This means the model was trained with different settings than `config.py` currently has. Delete `word_recog_model.keras`, `label_encoder.pkl`, and `norm_stats.npz`, then retrain.

---

## Platform Notes

- **Windows** — works out of the box. Falls back to `ctypes` for keypresses if `pyautogui` is unavailable.
- **macOS** — requires accessibility permissions for `pyautogui` (`System Preferences → Privacy → Accessibility`).
- **Linux** — requires `python3-tk` for the black screen feature (`sudo apt install python3-tk`).
