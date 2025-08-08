# visualizer/audio_utils.py

import os
import numpy as np
import librosa
from pydub import AudioSegment

def load_audio(path):
    """
    Load audio from WAV or M4A file and return waveform and sample rate.
    """
    ext = os.path.splitext(path)[1].lower()
    print(f"[load_audio] Loading {ext.upper()} file...")

    if ext == ".wav":
        y, sr = librosa.load(path, sr=None, mono=True)
    elif ext == ".m4a":
        audio = AudioSegment.from_file(path, format="m4a")
        audio = audio.set_channels(1).set_frame_rate(22050)  # librosa default
        y = np.array(audio.get_array_of_samples()).astype(np.float32)
        y /= np.iinfo(audio.array_type).max  # Normalize to [-1, 1]
        sr = audio.frame_rate
    else:
        raise ValueError("Unsupported format: use .wav or .m4a")

    return y, sr

def analyze_rhythm(y, sr, duration_limit=None):
    """
    Analyze tempo, beat, and normalized onset energy from audio waveform.
    """
    print("[analyze_rhythm] Analyzing rhythm structure...")

    if duration_limit:
        y = y[:int(duration_limit * sr)]

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    norm_energy = onset_env / np.max(onset_env)

    return {
        "tempo": tempo,
        "beat_frames": beat_frames,
        "onset_env": onset_env,
        "norm_energy": norm_energy,
        "times": times
    }