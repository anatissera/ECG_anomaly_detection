import os
import numpy as np
from IPython.display import Markdown, display
from typing import Optional


def find_data_subfolder(subfolder_name, start_path='.'):
    """Search for a subfolder inside the 'data' folder, walking up from start_path."""

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
    """Search for a folder with the given name.

    Walks up the parent directories first, so the notebooks resolve project
    folders correctly no matter which directory the kernel was started in, and
    only then falls back to a downward search from start_path.
    """

    current = os.path.abspath(start_path)

    # 1) direct child of start_path or of any of its ancestors
    node = current
    while True:
        candidate = os.path.join(node, folder_name)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent

    # 2) nested somewhere under the project root (nearest ancestor holding .git or src)
    node, project_root = current, None
    while True:
        if os.path.isdir(os.path.join(node, '.git')) or os.path.isdir(os.path.join(node, 'src')):
            project_root = node
            break
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent

    for base in (project_root, start_path):
        if not base:
            continue
        for root, dirs, _ in os.walk(base):
            if folder_name in dirs:
                return os.path.join(root, folder_name)
    return None

def display_metrics_as_md(metrics: dict, title: str = "Final Metrics on Unseen Test Set"):
    """Display metrics as Markdown."""

    md = f"<h2 style='margin-bottom:0.3em'>{title}</h2>\n"
    for name, value in metrics.items():
        pretty = name.replace('_', ' ').title()
        md += (
            f"<p style='font-size:16px; margin:0.2em 0'>"
            f"<strong>{pretty}:</strong> {value:.4f}"
            f"</p>\n"
        )
    display(Markdown(md))
