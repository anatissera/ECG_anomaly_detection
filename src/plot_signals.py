import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence, Dict

plt.rcParams.update({
    'figure.figsize': (10, 5),
    'axes.titlesize': 'big',
    'axes.labelsize': 'medium',
    'lines.linewidth': 2,
    'legend.fontsize': 'medium',
})

COLORS = {
    'normal': '#2ca02c',   # verde
    'anomaly': '#d62728'   # rojo
}

def subplot_ecg(ecg_data: np.ndarray, labels: Optional[Sequence[int]] = None, num_to_plot: int = 9, lead: int = 0, indices: Optional[Sequence[int]] = None):
    """Muestra subplots de señales ECG."""
    
    grid_size = int(np.sqrt(num_to_plot))
    assert grid_size**2 == num_to_plot, "num_to_plot debe ser un cuadrado perfecto"

    if indices is None:
        indices = np.random.choice(len(ecg_data), num_to_plot, replace=False)
    else:
        assert len(indices) == num_to_plot, "El número de índices debe coincidir con num_to_plot"

    for i, idx in enumerate(indices, start=1):
        ax = plt.subplot(grid_size, grid_size, i)
        sig = ecg_data[idx, :, lead] if ecg_data.ndim == 3 else ecg_data[idx]
        clase = labels[idx] if labels is not None else None
        color = COLORS['anomaly'] if clase and labels is not None else COLORS['normal']

        ax.plot(sig, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 110)

        title = f"Idx {idx}: {'Anómalo' if clase else 'Normal'}" if labels is not None else f"Idx {idx}"
        ax.set_title(title)

    plt.tight_layout()
    

MAIN_COLOR = '#4C72B0' 
WAVE_COLORS = {
    'P': '#1f77b4',  
    'Q': '#ff7f0e',  
    'R': '#2ca02c',  
    'S': '#d62728',  
    'T': '#9467bd'  
}
TITLE_BG = 'lavenderblush'  
THICK_LINE = 3
FONT_SIZES = {
    'title': 18,
    'labels': 16,
    'ticks': 14,
    'legend': 13
}

def plot_ecg_with_fmm(ecg: np.ndarray, coeffs: np.ndarray, convert_fn, num_leads: int, seq_len: int, fs: int = 100, lead: int = 0, label: str = None):
    """Dibuja señal ECG cruda y su reconstrucción FMM."""
    
    t = np.arange(seq_len) / fs

    # Señal original
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, ecg, color=MAIN_COLOR, linewidth=THICK_LINE, label=label or 'ECG original')
    ax.set_xlabel('Tiempo (s)', fontsize=FONT_SIZES['labels'])
    ax.set_ylabel('Amplitud', fontsize=FONT_SIZES['labels'])
    ax.tick_params(axis='both', labelsize=FONT_SIZES['ticks'])
    ax.set_xlim(0, 1.1)

    title = label or 'ECG cruda'
    ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.title.set_bbox(dict(facecolor=TITLE_BG, edgecolor='none', boxstyle='round,pad=0.5', alpha=0.7))
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(alpha=0.2)
    plt.tight_layout()

    # Reconstrucción FMM
    params = convert_fn(coeffs, num_leads=num_leads)
    waves = []
    for wave_name in ['P', 'Q', 'R', 'S', 'T']:
        wave = _generate_wave_segment(params[wave_name], seq_len, fs, lead)
        waves.append((wave_name, wave))

    reconstruccion = params['P']['M'][lead] + sum(w for _, w in waves)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, reconstruccion, color=MAIN_COLOR, linewidth=THICK_LINE, label='Reconstrucción FMM')
    for name, wave in waves:
        ax.plot(t, wave, linestyle='--', color=WAVE_COLORS[name], linewidth=2, label=f'Onda {name}')

    ax.set_xlabel('Tiempo (s)', fontsize=FONT_SIZES['labels'])
    ax.set_ylabel('Amplitud', fontsize=FONT_SIZES['labels'])
    ax.tick_params(axis='both', labelsize=FONT_SIZES['ticks'])
    ax.set_title('Reconstrucción FMM con ondas', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.title.set_bbox(dict(facecolor=TITLE_BG, edgecolor='none', boxstyle='round,pad=0.5', alpha=0.7))
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(alpha=0.2)
    plt.tight_layout()


def _generate_wave_segment(wave_dict: Dict[str, np.ndarray], seq_len: int, fs: int, lead: int) -> np.ndarray:
    """Genera segmento de onda FMM para un lead."""
    
    A = wave_dict['A'][lead]
    alpha = wave_dict['alpha'][0]
    beta = wave_dict['beta'][lead]
    omega = wave_dict['omega'][0]
    t = np.linspace(0, 2 * np.pi, seq_len)
    phase = beta + 2 * np.arctan(omega * np.tan((t - alpha) / 2))
    return A * np.cos(phase)
