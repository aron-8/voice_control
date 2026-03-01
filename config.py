# config.py
# Shared configuration for record.py, train.py, and recognize.py
# Edit values here — changes apply across the entire pipeline.

import numpy as np

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000       # Hz — must match your microphone capture rate
DURATION = 1.2              # seconds per clip (record & inference window)
CHUNK = 1024              # frames per sounddevice read block (~64ms at 16kHz)

# ---------------------------------------------------------------------------
# Words / classes
# ---------------------------------------------------------------------------
WORDS = [
    "forward",
    "back",
    "space",
    "down",
    "up",
    "dark",
    "exit",
    "lock",
    "noise",   # background noise class — always keep this
]

# ---------------------------------------------------------------------------
# Recording 
# ---------------------------------------------------------------------------
SAMPLES_PER_WORD = 50          # recordings to collect per word

# Voice activation
PREBUFFER_SEC = 0.5            # seconds of audio kept before trigger
CALIBRATE_SECONDS = 1.0        # duration of ambient noise measurement
THRESHOLD_MULTIPLIER = 1.5     # trigger when RMS > ambient * this value
ABSOLUTE_MIN = 0.005           # minimum trigger threshold (avoids zero ambient)

# ---------------------------------------------------------------------------
# MFCC features
# ---------------------------------------------------------------------------
N_MFCC = 13                    # number of MFCC coefficients per frame
HOP_LENGTH = 512               # librosa hop length (samples between frames)

# MAX_LEN: number of time frames in the MFCC matrix.
# Formula: ceil(SAMPLE_RATE * DURATION / HOP_LENGTH), with a small margin.
# At 16000 Hz, 1s duration, hop 512 → ~32 frames. We use 40 for safety.
MAX_LEN = int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH)) + 8  # = 40

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15               # fraction of data held out for validation
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 7

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data_1"
MODEL_PATH = "word_recog_model.keras"
ENCODER_PATH = "label_encoder.pkl"
NORM_PATH = "norm_stats.npz"   # mean & std saved during training for inference