import os
import numpy as np
from IPython.display import Markdown, display
from typing import Optional


def find_data_subfolder(subfolder_name, start_path='.'):
    """Busca subfolder dentro de carpeta data desde start_path."""
    
    current_path = os.path.abspath(start_path)
    while True:
        candidate = os.path.join(current_path, 'data', subfolder_name)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current_path)
        if parent == current_path:
            break
        current_path = parent
    return None

def find_folder(start_path: str = '.', folder_name: str = 'saved_models_and_params') -> Optional[str]:
    """Busca recursivamente una carpeta con nombre dado desde start_path."""
    
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

def display_metrics_as_md(metrics: dict, title: str = "Final Metrics on Unseen Test Set"):
    """Muestra métricas como Markdown."""
    
    md = f"<h2 style='margin-bottom:0.3em'>{title}</h2>\n"
    for name, value in metrics.items():
        pretty = name.replace('_', ' ').title()
        md += (
            f"<p style='font-size:16px; margin:0.2em 0'>"
            f"<strong>{pretty}:</strong> {value:.4f}"
            f"</p>\n"
        )
    display(Markdown(md))