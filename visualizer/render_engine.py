# visualizer/render_engine.py

import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

def init_plot():
    """
    Initializes the matplotlib figure and axis for visual rendering.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.style.use('dark_background')
    plt.tight_layout()
    return fig, ax

def make_frame_factory(y, sr, norm_energy, fig, ax, duration):
    """
    Returns a make_frame(t) function for MoviePy video rendering.
    
    Parameters:
    - y: waveform
    - sr: sample rate
    - norm_energy: onset strength, normalized [0–1]
    - fig, ax: matplotlib figure/axis
    - duration: audio duration in seconds
    """

    def make_frame(t):
        idx = int(t * sr)
        window = y[idx:idx + 2048]
        if len(window) < 2048:
            window = np.pad(window, (0, 2048 - len(window)))

        fft = np.abs(np.fft.rfft(window))[:512]
        fft = fft / np.max(fft) if np.max(fft) > 0 else fft

        energy_idx = min(int(t * len(norm_energy) / duration), len(norm_energy) - 1)
        energy = norm_energy[energy_idx]

        # === RENDER ===
        ax.clear()
        ax.plot(fft, color='cyan', lw=2)
        ax.fill_between(range(len(fft)), fft, color='cyan', alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(fft))
        ax.axis('off')
        ax.set_title("AstroEnergies Visualizer", fontsize=12, color='white', pad=10)

        # Pulse background with beat energy
        fig.patch.set_alpha(0.6 + 0.4 * energy)

        return mplfig_to_npimage(fig)

    return make_frame