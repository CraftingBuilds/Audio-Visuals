# main.py — AstroEnergies Visualizer (Modular Build)

import os
import librosa
import moviepy.editor as mpy
from visualizer import (
    load_audio,
    analyze_rhythm,
    init_plot,
    make_frame_factory
)

# === CONFIG ===
AUDIO_PATH = "assets/your_audio.m4a"  # .wav or .m4a
OUTPUT_PATH = "output/generated_video.mp4"
FPS = 30
DURATION_LIMIT = 60  # seconds (for test render)

# === STEP 1: Load Audio ===
y, sr = load_audio(AUDIO_PATH)
duration = min(librosa.get_duration(y=y, sr=sr), DURATION_LIMIT)

# === STEP 2: Analyze Beat & Rhythm ===
rhythm = analyze_rhythm(y, sr, duration_limit=DURATION_LIMIT)
norm_energy = rhythm["norm_energy"]

# === STEP 3: Initialize Rendering ===
fig, ax = init_plot()
make_frame = make_frame_factory(y, sr, norm_energy, fig, ax, duration)

# === STEP 4: Render Video ===
print(f"[Rendering] {OUTPUT_PATH}")
clip = mpy.VideoClip(make_frame, duration=duration)
audio_clip = mpy.AudioFileClip(AUDIO_PATH).subclip(0, duration)
final = clip.set_audio(audio_clip)

final.write_videofile(OUTPUT_PATH, fps=FPS, audio_codec="aac")

print("✅ Done: Video saved to:", OUTPUT_PATH)