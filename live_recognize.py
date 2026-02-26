# live_recognize.py
import sounddevice as sd
import numpy as np
import joblib
import librosa
import time
from collections import deque
from tensorflow.keras.models import load_model
import threading
import platform
import sys
import ctypes
import tkinter as tk

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False

if platform.system() == 'Windows' and not PYAUTOGUI_AVAILABLE:
    import ctypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)

from config import (
    SAMPLE_RATE, DURATION, CHUNK, N_MFCC, MAX_LEN, HOP_LENGTH,
    PREBUFFER_SEC, CALIBRATE_SECONDS, THRESHOLD_MULTIPLIER, ABSOLUTE_MIN,
    MODEL_PATH, ENCODER_PATH, NORM_PATH
)

# Confidence threshold — predictions below this are ignored
CONF_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Load model, encoder, and normalization stats (must match train.py exactly)
# ---------------------------------------------------------------------------
print("Loading model...")
model = load_model(MODEL_PATH)
le    = joblib.load(ENCODER_PATH)

_norm        = np.load(NORM_PATH)
NORM_MEAN    = _norm['mean']   # shape: (1, 1, N_MFCC)
NORM_STD     = _norm['std']    # shape: (1, 1, N_MFCC)

print(f"Classes: {list(le.classes_)}")


# ---------------------------------------------------------------------------
# Black screen manager (Tkinter, runs in background thread)
# ---------------------------------------------------------------------------_tk_root = None
_tk_windows = []  # one window per monitor

def init_tk():
    global _tk_root, _tk_windows

    import screeninfo
    monitors = screeninfo.get_monitors()
    print(f"Detected {len(monitors)} monitors:")
    for m in monitors:
        print(f"  {m.width}x{m.height}+{m.x}+{m.y}")

    # Create root window first (required by Tkinter)
    _tk_root = tk.Tk()
    _tk_root.configure(background='black')
    _tk_root.withdraw()  # start hidden, we'll manage visibility manually

    _tk_windows.clear()

    for i, m in enumerate(monitors):
        if i == 0:
            win = _tk_root
        else:
            win = tk.Toplevel(_tk_root)
            win.configure(background='black')

        # Position and size exactly on this monitor
        win.geometry(f"{m.width}x{m.height}+{m.x}+{m.y}")
        win.attributes('-topmost', True)
        win.overrideredirect(True)  # removes title bar and window borders
        win.withdraw()
        _tk_windows.append(win)

def show_black_screen():
    for win in _tk_windows:
        win.deiconify()
        win.lift()

def hide_black_screen():
    for win in _tk_windows:
        win.withdraw()
        
def lock_workstation():
    """lock the workstation"""
    ctypes.windll.user32.LockWorkStation()

# ---------------------------------------------------------------------------
# Key sending
# ---------------------------------------------------------------------------

def send_key(key_name: str):
    """Send a key press via pyautogui or Windows ctypes fallback."""
    if PYAUTOGUI_AVAILABLE:
        try:
            pyautogui.press(key_name)
            return
        except Exception:
            pass

    if platform.system() == 'Windows':
        key_map = {
            'space': 0x20, 'left': 0x25, 'up': 0x26,
            'right': 0x27, 'down': 0x28,
        }
        vk = key_map.get(key_name.lower())
        if vk is None and len(key_name) == 1:
            vk = ord(key_name.upper())
        if not vk:
            print(f"[send_key] Unknown key: '{key_name}'")
            return
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(vk, 0, 2, 0)
    else:
        print(f"[send_key] No fallback available for key '{key_name}' on this platform.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def rms(frames: np.ndarray) -> float:
    if frames.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frames.astype(np.float64)))))


def extract_features(audio: np.ndarray) -> np.ndarray:
    """
    Convert raw 1-D waveform → normalized MFCC ready for model inference.
    Must mirror the exact pipeline used in train.py.
    """
    target = int(SAMPLE_RATE * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).T
    # shape: (frames, N_MFCC)

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]
    # shape: (MAX_LEN, N_MFCC)

    # Apply same normalization as training
    mean = NORM_MEAN.squeeze()   # (1,1,13) → (13 ,)
    std  = NORM_STD.squeeze()    # (1,1,13) → (13,)
    mfcc = (mfcc - mean) / (std + 1e-8)
    # shape stays clean: (MAX_LEN, N_MFCC)

    # Add batch + channel dims → (1, MAX_LEN, N_MFCC, 1)
    return mfcc[np.newaxis, ..., np.newaxis]


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

def dispatch(label: str):
    """Map a recognized label to an OS action."""
    lab = label.lower()

    if lab == 'noise':
        return  # background noise — do nothing

    action_map = {
        'forward':  lambda: send_key('right'),
        'back':     lambda: send_key('left'),
        'space':     lambda: send_key('space'),
        'play':     lambda: send_key('space'),
        'down':    lambda: send_key('down'),
        'up':   lambda: send_key('up'),
        'lock':     lambda: lock_workstation(),
        'dark':     lambda: show_black_screen(),
        'exit':     lambda: hide_black_screen(),
    }
 
    action = action_map.get(lab)
    if action:
        action()
    else:
        print(f"[dispatch] No action mapped for label: '{label}'")


# ---------------------------------------------------------------------------
# Recognition loop
# ---------------------------------------------------------------------------
def audio_loop():
    """Full recognition loop — runs in background thread."""

    print("Live recognition started. Press Ctrl+C to quit.")

    target_frames  = int(DURATION * SAMPLE_RATE)
    prechunk_count = int(np.ceil(PREBUFFER_SEC * SAMPLE_RATE / CHUNK))

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK) as stream:

            # Ambient calibration
            print("Calibrating ambient noise — stay silent...")
            calib_chunks   = max(1, int(CALIBRATE_SECONDS * SAMPLE_RATE / CHUNK))
            ambient_levels = []
            for _ in range(calib_chunks):
                frames, _ = stream.read(CHUNK)
                ambient_levels.append(rms(frames[:, 0]))
            ambient   = float(np.mean(ambient_levels))
            threshold = max(ABSOLUTE_MIN, ambient * THRESHOLD_MULTIPLIER)
            print(f"Ambient RMS: {ambient:.6f}  |  Trigger threshold: {threshold:.6f}")
            print("Listening...\n")

            prebuf = deque(maxlen=prechunk_count)

            while True:
                frames, _ = stream.read(CHUNK)
                frames = frames[:, 0]
                prebuf.append(frames.copy())

                if rms(frames) < threshold:
                    continue

                print(f"Trigger detected — recognizing...")

                # Collect audio: prebuffer + enough new chunks to fill DURATION
                collected = list(prebuf)
                total     = sum(a.shape[0] for a in collected)

                while total < target_frames:
                    frames2, _ = stream.read(CHUNK)
                    frames2 = frames2[:, 0]
                    collected.append(frames2.copy())
                    total += frames2.shape[0]

                # Trim from front to exact length
                audio = np.concatenate(collected, axis=0)[:target_frames]

                # Predict
                x     = extract_features(audio)
                probs = model.predict(x, verbose=0)[0]
                idx   = int(np.argmax(probs))
                label = le.inverse_transform([idx])[0]
                conf  = float(probs[idx])

                if conf >= CONF_THRESHOLD:
                    print(f"Recognized: '{label}'  (confidence: {conf:.2f})")
                    dispatch(label)
                else:
                    runner_up = le.inverse_transform([np.argsort(probs)[-2]])[0]
                    print(f"Low confidence — best: '{label}' ({conf:.2f}), runner-up: '{runner_up}' ({probs[np.argsort(probs)[-2]]:.2f})")

                # Cooldown to prevent re-triggering on the same utterance
                time.sleep(0.3)
                prebuf.clear()

    except KeyboardInterrupt:
        print("\nExiting.")
        hide_black_screen()
        sys.exit(0)


def main():
    init_tk()

    # Start audio recognition in background thread
    t = threading.Thread(target=audio_loop, daemon=True)
    t.start()

    # Tkinter owns the main thread — checks every 100ms if thread is alive
    def check_alive():
        if t.is_alive():
            _tk_root.after(100, check_alive)
        else:
            _tk_root.destroy()

    _tk_root.after(100, check_alive)

    try:
        _tk_root.mainloop()
    except KeyboardInterrupt:
        hide_black_screen()


if __name__ == '__main__':
    main()