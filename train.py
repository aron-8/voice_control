# train.py
import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from config import (
    SAMPLE_RATE, DURATION, N_MFCC, MAX_LEN, HOP_LENGTH,
    DATA_DIR, MODEL_PATH, ENCODER_PATH, NORM_PATH,
    TEST_SIZE, RANDOM_STATE, EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE
)

# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def augment_waveform(wav: np.ndarray) -> list[np.ndarray]:
    """Return a list of augmented copies of a waveform."""
    augmented = []

    # 1. Add Gaussian noise
    noise = wav + 0.005 * np.random.randn(len(wav))
    augmented.append(noise.astype(np.float32))

    # 2. Time shift (±20% of length)
    shift = int(SAMPLE_RATE * DURATION * 0.2)
    shift_amount = np.random.randint(-shift, shift)
    shifted = np.roll(wav, shift_amount)
    # zero out the wrapped region
    if shift_amount > 0:
        shifted[:shift_amount] = 0.0
    else:
        shifted[shift_amount:] = 0.0
    augmented.append(shifted.astype(np.float32))

    # 3. Pitch shift (±2 semitones)
    n_steps = np.random.uniform(-2, 2)
    pitched = librosa.effects.pitch_shift(wav, sr=SAMPLE_RATE, n_steps=n_steps)
    augmented.append(pitched.astype(np.float32))

    # 4. Time stretch (speed up or slow down slightly)
    rate = np.random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(wav, rate=rate)
    # re-pad/trim to fixed length after stretching
    target = int(SAMPLE_RATE * DURATION)
    if len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)))
    else:
        stretched = stretched[:target]
    augmented.append(stretched.astype(np.float32))

    # 5. Volume scaling
    scale = np.random.uniform(0.7, 1.3)
    scaled = (wav * scale).astype(np.float32)
    augmented.append(scaled)

    return augmented


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def wav_to_mfcc(wav: np.ndarray) -> np.ndarray:
    """Convert a fixed-length waveform to a padded/trimmed MFCC array."""
    mfcc = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mf = mfcc.T  # shape: (frames, N_MFCC)
    if mf.shape[0] < MAX_LEN:
        mf = np.pad(mf, ((0, MAX_LEN - mf.shape[0]), (0, 0)))
    else:
        mf = mf[:MAX_LEN, :]
    return mf  # shape: (MAX_LEN, N_MFCC)


def load_files(data_dir: str, augment: bool = True):
    """Load all wav files, optionally augmenting each sample."""
    X, y = [], []
    target_len = int(SAMPLE_RATE * DURATION)

    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        wav_files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
        if not wav_files:
            continue

        print(f"  '{label}': {len(wav_files)} original files", end="")

        for fname in wav_files:
            path = os.path.join(label_dir, fname)
            wav, _ = librosa.load(path, sr=SAMPLE_RATE)

            # Pad / trim to exact length
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            else:
                wav = wav[:target_len]
            wav = wav.astype(np.float32)

            # Original sample
            X.append(wav_to_mfcc(wav))
            y.append(label)

            # Augmented samples
            if augment:
                for aug_wav in augment_waveform(wav):
                    X.append(wav_to_mfcc(aug_wav))
                    y.append(label)

        total = len(wav_files) * (6 if augment else 1)  # 1 original + 5 augmented
        print(f" → {total} total (with augmentation)" if augment else "")

    return np.array(X, dtype=np.float32), np.array(y)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(X: np.ndarray, mean=None, std=None):
    """
    Normalize MFCC features per coefficient (across time and samples).
    If mean/std not provided, compute from X (training set).
    Returns normalized X, mean, std.
    """
    if mean is None:
        mean = X.mean(axis=(0, 1), keepdims=True)  # shape: (1, 1, N_MFCC)
        std  = X.std(axis=(0, 1), keepdims=True)
    return (X - mean) / (std + 1e-8), mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_shape: tuple, num_classes: int) -> models.Sequential:
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n=== Loading and augmenting data ===")
    X, y = load_files(DATA_DIR, augment=False)
    print(f"\nDataset shape: {X.shape}, labels: {y.shape}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_enc, num_classes=num_classes)
    print(f"Classes ({num_classes}): {list(le.classes_)}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cat
    )

    # Normalize — fit on train only, apply to both
    print("\n=== Normalizing ===")
    X_train, mean, std = normalize(X_train)
    X_test, _, _       = normalize(X_test, mean=mean, std=std)

    # Save normalization stats for inference
    np.savez(NORM_PATH, mean=mean, std=std)
    print(f"Normalization stats saved → {NORM_PATH}")

    # Add channel dim for CNN: (samples, MAX_LEN, N_MFCC) → (samples, MAX_LEN, N_MFCC, 1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    # Build & summarize model
    print("\n=== Model ===")
    input_shape = X_train.shape[1:]  # (MAX_LEN, N_MFCC, 1)
    model = build_model(input_shape, num_classes)
    model.summary()

    # Train
    print("\n=== Training ===")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

    # Save model, encoder, norm stats
    model.save(MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\nSaved: {MODEL_PATH}, {ENCODER_PATH}, {NORM_PATH}")