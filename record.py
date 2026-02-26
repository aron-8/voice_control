# record.py
import sounddevice as sd
import soundfile as sf
import os
import time
import numpy as np
from collections import deque

from config import (
    SAMPLE_RATE, DURATION, CHUNK, WORDS, SAMPLES_PER_WORD,
    PREBUFFER_SEC, CALIBRATE_SECONDS, THRESHOLD_MULTIPLIER, ABSOLUTE_MIN,
    DATA_DIR
)

NOISE_LABEL = "noise"   # clips of this word are recorded without a trigger

os.makedirs(DATA_DIR, exist_ok=True)


def rms(frames: np.ndarray) -> float:
    """Return RMS of a 1-D numpy array."""
    if frames.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frames.astype(np.float64)))))


def _next_index(word_dir: str) -> int:
    """Return the next sample index based on existing .wav files."""
    existing = [f for f in os.listdir(word_dir) if f.endswith(".wav")]
    return len(existing) + 1


def record_noise_samples(word: str):
    """
    Record background noise samples continuously — no voice-activation threshold.
    Each DURATION-second chunk is saved as a separate file automatically.
    """
    d = os.path.join(DATA_DIR, word)
    os.makedirs(d, exist_ok=True)
    start_index = _next_index(d)
    target_frames = int(DURATION * SAMPLE_RATE)

    print(f"\n--- Noise recording ---")
    print(f"Recording {SAMPLES_PER_WORD} ambient noise samples. Stay quiet / make typical background noise.")
    print("Recording starts immediately — no trigger needed.\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK) as stream:
            sample_num = start_index
            while sample_num < start_index + SAMPLES_PER_WORD:
                collected = []
                total = 0
                # Read exactly target_frames worth of audio
                while total < target_frames:
                    frames, _ = stream.read(CHUNK)
                    frames = frames[:, 0]
                    collected.append(frames.copy())
                    total += frames.shape[0]

                audio = np.concatenate(collected, axis=0)[:target_frames]

                fname = os.path.join(d, f"{word}_{sample_num:03d}.wav")
                sf.write(fname, audio, SAMPLE_RATE)
                print(f"Saved noise sample: {fname}")
                sample_num += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Recording error: {e}")


def record_word_samples(word: str):
    """
    Record samples for a keyword using voice-activation (RMS threshold trigger).
    Includes a pre-buffer so the start of the word is never clipped.
    """
    d = os.path.join(DATA_DIR, word)
    os.makedirs(d, exist_ok=True)
    start_index = _next_index(d)
    target_frames = int(DURATION * SAMPLE_RATE)
    prechunk_count = int(np.ceil(PREBUFFER_SEC * SAMPLE_RATE / CHUNK))

    print(f"\n--- '{word}' ---")
    print(f"Recording {SAMPLES_PER_WORD} samples, starting at index {start_index}.")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK) as stream:

            # Ambient noise calibration
            print("Calibrating ambient noise — stay silent...")
            calib_chunks = max(1, int(CALIBRATE_SECONDS * SAMPLE_RATE / CHUNK))
            ambient_levels = []
            for _ in range(calib_chunks):
                frames, _ = stream.read(CHUNK)
                ambient_levels.append(rms(frames[:, 0]))
            ambient = float(np.mean(ambient_levels))
            threshold = max(ABSOLUTE_MIN, ambient * THRESHOLD_MULTIPLIER)
            print(f"Ambient RMS: {ambient:.6f}  |  Trigger threshold: {threshold:.6f}")
            print("Speak now when ready...\n")

            prebuf = deque(maxlen=prechunk_count)
            sample_num = start_index

            while sample_num < SAMPLES_PER_WORD:
                frames, _ = stream.read(CHUNK)
                frames = frames[:, 0]
                prebuf.append(frames.copy())

                if rms(frames) >= threshold:
                    print(f"Trigger detected — recording sample {sample_num}...")

                    # Start from pre-buffer (captures onset of the word)
                    collected = list(prebuf)
                    total = sum(a.shape[0] for a in collected)

                    # Read remaining frames in full CHUNK increments
                    while total < target_frames:
                        frames2, _ = stream.read(CHUNK)
                        frames2 = frames2[:, 0]
                        collected.append(frames2.copy())
                        total += frames2.shape[0]

                    # Trim to exact length from the front
                    audio = np.concatenate(collected, axis=0)[:target_frames]

                    fname = os.path.join(d, f"{word}_{sample_num:03d}.wav")
                    sf.write(fname, audio, SAMPLE_RATE)
                    print(f"Saved: {fname}\n")
                    sample_num += 1

                    # Brief pause to avoid re-triggering on the same utterance
                    time.sleep(0.2)
                    prebuf.clear()

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Recording error: {e}")


def main():
    for word in WORDS:
        if word == NOISE_LABEL:
            record_noise_samples(word)
        else:
            record_word_samples(word)


if __name__ == '__main__':
    main()