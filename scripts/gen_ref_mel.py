#!/usr/bin/env python3
"""
Generate a reference log-mel spectrogram JSON file for validating the Go
audio preprocessing implementation against NeMo/librosa.

Output: audio/testdata/ref_mel_htk.json

Usage:
    uv run python3 scripts/gen_ref_mel.py
"""

import json
import math
import numpy as np

try:
    import librosa
except ImportError:
    raise SystemExit("Install librosa: pip install librosa")


def gen_test_signal(sr=16000, duration=1.0):
    """Deterministic test signal: sum of sines at several frequencies."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (
        0.5 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
        + 0.15 * np.sin(2 * np.pi * 3520 * t)
    ).astype(np.float32)
    return sig


def preemphasis(sig, coeff=0.97):
    """NeMo-style pre-emphasis: y[0]=x[0], y[n]=x[n]-coeff*x[n-1] for n>0.
    Matches torch.cat((x[:,0:1], x[:,1:] - preemph*x[:,:-1]), dim=1) in NeMo.
    """
    out = np.zeros_like(sig, dtype=np.float64)
    out[0] = sig[0]  # NeMo keeps y[0] = x[0] unchanged
    for i in range(1, len(sig)):
        out[i] = sig[i] - coeff * sig[i - 1]
    return out


def log_mel_nemo(
    pcm,
    sr=16000,
    window_size=0.02,
    window_stride=0.01,
    n_fft=512,
    n_mels=80,
    fmax=8000.0,
    preemph=0.97,
    log_offset=1e-6,
):
    """
    Replicate NeMo's AudioToMelSpectrogramPreprocessor (no normalization).
    NeMo defaults: center=True, mel_norm="slaney" (Slaney scale + area norm),
    periodic Hann window, power spectrum.
    """
    win_len = int(round(window_size * sr))
    hop_len = int(round(window_stride * sr))

    sig = preemphasis(pcm.astype(np.float64), preemph).astype(np.float32)

    # NeMo uses torch.hann_window(win_length, periodic=True), which is the
    # DFT-even (periodic) Hann window: w[i] = 0.5*(1 - cos(2*pi*i/N)).
    # librosa defaults to the symmetric window (w[N-1]=0), so we must pass
    # the periodic window explicitly to match NeMo.
    periodic_hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_len) / win_len))

    # STFT: center=True + constant padding matches NeMo's default behavior.
    stft = librosa.stft(
        sig,
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=win_len,
        window=periodic_hann,
        center=True,
        pad_mode="constant",
    )
    power = np.abs(stft) ** 2  # [n_fft/2+1, n_frames]

    # Mel filterbank: NeMo default is mel_norm="slaney".
    # librosa: htk=False (default, Slaney Hz scale) + norm="slaney" (area norm).
    mel_fb = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=fmax, htk=False, norm="slaney"
    )
    mel_spec = mel_fb @ power  # [n_mels, n_frames]
    log_mel = np.log(mel_spec + log_offset)  # [n_mels, n_frames]

    return log_mel.astype(np.float32)


def main():
    cfg = {
        "sample_rate": 16000,
        "window_size": 0.02,
        "window_stride": 0.01,
        "n_fft": 512,
        "n_mels": 80,
        "fmax": 8000.0,
        "preemph": 0.97,
    }

    pcm = gen_test_signal(sr=cfg["sample_rate"])
    log_mel = log_mel_nemo(
        pcm,
        sr=cfg["sample_rate"],
        window_size=cfg["window_size"],
        window_stride=cfg["window_stride"],
        n_fft=cfg["n_fft"],
        n_mels=cfg["n_mels"],
        fmax=cfg["fmax"],
        preemph=cfg["preemph"],
    )

    n_frames = log_mel.shape[1]

    out = {
        "config": cfg,
        "pcm": pcm.tolist(),
        "log_mel": log_mel.flatten().tolist(),  # row-major [n_mels × n_frames]
        "n_frames": int(n_frames),
    }

    import os
    os.makedirs("audio/testdata", exist_ok=True)
    path = "audio/testdata/ref_mel_htk.json"
    with open(path, "w") as f:
        json.dump(out, f)

    print(f"Wrote {path}")
    print(f"  PCM samples : {len(pcm)}")
    print(f"  n_frames    : {n_frames}")
    print(f"  log_mel shape: [{cfg['n_mels']} × {n_frames}]")
    print(f"  log_mel min/max: {log_mel.min():.4f} / {log_mel.max():.4f}")


if __name__ == "__main__":
    main()
